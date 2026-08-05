from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .prompts import Prompts


class AnswerGenerator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        runtime = config.get("runtime", {})
        self.backend = str(runtime.get("answer_backend", "transformers"))
        self.model = None
        self.tokenizer = None
        if self.backend == "transformers":
            self._load_transformers()
        elif self.backend != "deterministic":
            raise ValueError(f"Unsupported runtime.answer_backend: {self.backend}")

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm = self.config.get("llm", {})
        model_path = llm.get("llm_path")
        if not model_path:
            raise ValueError("llm.llm_path is required for the transformers answer backend")
        self.device = str(llm.get("llm_device", "cuda"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else None,
        )
        self.model.eval()
        self.model.to(self.device)

    def generate(
        self,
        *,
        dataset_name: str,
        question: str,
        evidence: str,
    ) -> Tuple[str, Optional[Dict[str, float]], str]:
        if self.backend == "deterministic":
            predictions = self.config.get("runtime", {}).get("deterministic_predictions", {})
            prediction = predictions.get(question, self.config.get("runtime", {}).get("deterministic_default", "A"))
            mode = "deterministic_test_backend"
            return str(prediction), None, mode
        if dataset_name in {"InfiniteChoice", "NovelQA"}:
            return self._multiple_choice(question, evidence)
        return self._open_qa(question, evidence)

    def _multiple_choice(self, question: str, evidence: str) -> Tuple[str, Dict[str, float], str]:
        import torch

        prompt = Prompts["QA_prompt_options"].format(question=question, evidence=evidence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        labels = ["A", "B", "C", "D"]
        label_logits = torch.tensor(
            [logits[self.tokenizer(label).input_ids[-1]] for label in labels],
            dtype=torch.float32,
        )
        probabilities = torch.nn.functional.softmax(label_logits, dim=0).detach().cpu().tolist()
        best = labels[max(range(len(labels)), key=lambda index: probabilities[index])]
        return best, dict(zip(labels, probabilities)), "multiple_choice_logits"

    def _open_qa(self, question: str, evidence: str) -> Tuple[str, None, str]:
        import torch

        language = str(self.config.get("extractor", {}).get("language", "en"))
        prompt_name = "QA_prompt_answer_zh" if language == "zh" else "QA_prompt_answer"
        prompt = Prompts[prompt_name].format(question=question, evidence=evidence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(self.config.get("runtime", {}).get("max_new_tokens", 300)),
                do_sample=False,
            )
        generated = output[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), None, "open_qa_generate"
