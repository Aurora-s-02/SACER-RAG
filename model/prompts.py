Prompts = {
    "QA_prompt_options": """You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be an option among "A", "B", and "C", and "D" that supported by the given evidences and matches the question. You should not assume any information beyond the evidence. You should only output the option.

Question: {question}
Evidence: {evidence}

Answer: """,
    "QA_prompt_answer": """You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be a short sentence that is supported by the given evidences and matches the requirements of the question. You should not assume any information beyond the evidence. You should only output the answer.

Question: {question}
Evidence: {evidence}

Answer: """,
    "QA_prompt_answer_zh": """你是一个根据证据回答问题的助手。请只根据给定证据回答问题，不要假设证据之外的信息。答案应是简短句子。

问题：{question}
证据：{evidence}

答案：""",
}

