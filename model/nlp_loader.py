from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any, Dict, List, Literal


logger = logging.getLogger(__name__)


def load_nlp(language: str = "en", method: Literal["Spacy", "NLTK", "BERT_NER_POS"] = "Spacy"):
    if method == "Spacy":
        return SpacyExtractor(language)
    if method == "NLTK":
        return NLTKExtractor(language)
    if method == "BERT_NER_POS":
        return RegexExtractor(language, method="BERT_NER_POS")
    return RegexExtractor(language, method=str(method))


class RegexExtractor:
    def __init__(self, language: str = "en", method: str = "Regex"):
        self.language = language
        self.method = method

    def naive_extract_graph(self, text: str) -> Dict[str, Any]:
        terms = []
        seen = set()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", str(text or "")):
            if len(token) < 2:
                continue
            if token[0].isupper() or len(token) > 4:
                normalized = token if token[0].isupper() else token.lower()
                if normalized not in seen:
                    seen.add(normalized)
                    terms.append(normalized)
        pairs = {tuple(sorted(pair)): 1 for pair in combinations(terms, 2)}
        return {
            "nouns": terms,
            "cooccurrence": pairs,
            "double_nouns": {},
            "appearance_count": {term: 1 for term in terms},
        }


class SpacyExtractor(RegexExtractor):
    def __init__(self, language: str = "en"):
        super().__init__(language, method="Spacy")
        self.nlp = self._load_model(language)

    def _load_model(self, language: str):
        try:
            import spacy

            model_name = "en_core_web_lg" if language in {"en", "zh"} else "en_core_web_lg"
            return spacy.load(model_name)
        except Exception as exc:
            logger.warning("Falling back to regex extractor because spaCy is unavailable: %s", exc)
            return None

    def naive_extract_graph(self, text: str) -> Dict[str, Any]:
        if self.nlp is None:
            return super().naive_extract_graph(text)
        doc = self.nlp(str(text or ""))
        terms: List[str] = []
        seen = set()
        appearance_count: Dict[str, int] = {}
        for sent in doc.sents:
            sentence_terms: List[str] = []
            entity_token_ids = set()
            for ent in sent.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE"}:
                    pieces = ent.text.split()
                    values = pieces if ent.label_ == "PERSON" and len(pieces) >= 2 else [ent.text]
                    for value in values:
                        if value not in seen:
                            seen.add(value)
                            terms.append(value)
                        sentence_terms.append(value)
                        appearance_count[value] = appearance_count.get(value, 0) + 1
                for token in ent:
                    entity_token_ids.add(token.i)
            for token in sent:
                if token.i in entity_token_ids:
                    continue
                if token.pos_ in {"NOUN", "PROPN"}:
                    value = token.lemma_.lower() if token.pos_ == "NOUN" else token.lemma_.lower()
                    if value.strip():
                        if value not in seen:
                            seen.add(value)
                            terms.append(value)
                        sentence_terms.append(value)
                        appearance_count[value] = appearance_count.get(value, 0) + 1
        pairs = {tuple(sorted(pair)): 1 for pair in combinations(terms, 2)}
        return {
            "nouns": terms,
            "cooccurrence": pairs,
            "double_nouns": {},
            "appearance_count": appearance_count,
        }


class NLTKExtractor(RegexExtractor):
    def __init__(self, language: str = "en"):
        super().__init__(language, method="NLTK")

    def naive_extract_graph(self, text: str) -> Dict[str, Any]:
        try:
            import nltk

            terms: List[str] = []
            seen = set()
            appearance_count: Dict[str, int] = {}
            for sentence in nltk.tokenize.sent_tokenize(str(text or "")):
                tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
                for word, pos in tagged:
                    if pos.startswith("NN"):
                        value = word if pos.startswith("NNP") else word.lower()
                        if value not in seen:
                            seen.add(value)
                            terms.append(value)
                        appearance_count[value] = appearance_count.get(value, 0) + 1
            pairs = {tuple(sorted(pair)): 1 for pair in combinations(terms, 2)}
            return {
                "nouns": terms,
                "cooccurrence": pairs,
                "double_nouns": {},
                "appearance_count": appearance_count,
            }
        except Exception as exc:
            logger.warning("Falling back to regex extractor because NLTK is unavailable: %s", exc)
            return super().naive_extract_graph(text)
