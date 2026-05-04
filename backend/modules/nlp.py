"""NLP preprocessing and academic-domain validation."""

from __future__ import annotations

import re
from dataclasses import dataclass


ACADEMIC_KEYWORDS = {
    "academic", "algebra", "algorithm", "algorithms", "analysis", "biology", "calculus",
    "chemistry", "computer", "concept", "database", "derivative", "economics",
    "engineering", "exam", "explain", "flashcard", "flashcards", "formula", "gradient", "history",
    "learning", "literature", "machine", "math", "network", "physics",
    "programming", "quiz", "research", "science", "statistics", "study",
    "theory", "university",
}

NON_ACADEMIC_HINTS = {
    "movie", "relationship", "weather", "song", "shopping", "restaurant",
    "dating", "celebrity", "sports score", "stock price",
}

STOP_PHRASES = [
    r"\bplease\b", r"\bcan you\b", r"\bcould you\b", r"\bexplain\b",
    r"\bteach\b", r"\bmake\b", r"\bcreate\b", r"\bgenerate\b",
    r"\bgive me\b", r"\bshow me\b", r"\bstep by step\b", r"\bquiz\b",
    r"\bmcqs?\b", r"\bflashcards?\b", r"\bflash cards?\b", r"\bon\b",
    r"\babout\b", r"\bfor\b",
]


@dataclass(frozen=True)
class ProcessedQuery:
    raw: str
    cleaned: str
    topic: str
    tokens: list[str]
    mode: str
    is_academic: bool
    rejection_reason: str | None = None


class NLPProcessor:
    """Normalize user queries before retrieval and generation."""

    def preprocess(self, query: str) -> ProcessedQuery:
        raw = query.strip()
        cleaned = re.sub(r"\s+", " ", raw)
        mode = self.detect_mode(cleaned)
        topic = self.extract_topic(cleaned)
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*", cleaned.lower())
        is_academic = self.is_academic(cleaned, tokens)
        reason = None if is_academic else "Only academic and study-related questions are supported."

        return ProcessedQuery(
            raw=raw,
            cleaned=cleaned,
            topic=topic,
            tokens=tokens,
            mode=mode,
            is_academic=is_academic,
            rejection_reason=reason,
        )

    def detect_mode(self, query: str) -> str:
        query_lower = query.lower()
        if any(term in query_lower for term in ["quiz", "mcq", "multiple choice", "test me"]):
            return "quiz"
        if any(term in query_lower for term in ["flashcard", "flash card", "study card"]):
            return "flashcards"
        return "explanation"

    def extract_topic(self, query: str) -> str:
        topic = query.lower()
        for phrase in STOP_PHRASES:
            topic = re.sub(phrase, " ", topic)
        topic = re.sub(r"[^a-zA-Z0-9+\- ]+", " ", topic)
        topic = re.sub(r"\s+", " ", topic).strip()
        return topic or query.strip()

    def is_academic(self, query: str, tokens: list[str]) -> bool:
        lowered = query.lower()
        if not query.strip():
            return False
        if any(hint in lowered for hint in NON_ACADEMIC_HINTS):
            return False
        return any(token in ACADEMIC_KEYWORDS for token in tokens) or "what is" in lowered or "define" in lowered
