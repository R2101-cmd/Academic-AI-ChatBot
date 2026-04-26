"""Query validation utilities."""

from __future__ import annotations

import re

def is_academic_query(query: str) -> bool:
    """
    Accept any non-empty query.

    The assistant is now allowed to respond to broad topics, so this check
    only guards against empty input while preserving the existing pipeline API.
    """
    return bool(query and query.strip())


def detect_request_mode(query: str) -> str:
    """Detect whether the student asked for explanation, flashcards, or quiz."""
    query_lower = query.lower()

    quiz_terms = [
        "quiz",
        "mcq",
        "multiple choice",
        "practice question",
        "test me",
    ]
    flashcard_terms = [
        "flashcard",
        "flash card",
        "study card",
        "revision card",
    ]

    if any(term in query_lower for term in quiz_terms):
        return "quiz"
    if any(term in query_lower for term in flashcard_terms):
        return "flashcards"
    return "explanation"


def extract_topic_query(query: str) -> str:
    """Remove explicit study-mode phrases so retrieval focuses on the topic."""
    cleaned = query.lower()
    phrases = [
        r"\bmake\b",
        r"\bgenerate\b",
        r"\bcreate\b",
        r"\bgive me\b",
        r"\bshow me\b",
        r"\bflashcards?\b",
        r"\bflash cards?\b",
        r"\bstudy cards?\b",
        r"\bquiz\b",
        r"\bmcqs?\b",
        r"\bmultiple choice questions?\b",
        r"\bmultiple choice\b",
        r"\bpractice questions?\b",
        r"\btest me\b",
        r"\bon\b",
        r"\babout\b",
        r"\bfor\b",
    ]
    for phrase in phrases:
        cleaned = re.sub(phrase, " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?,.")
    return cleaned or query.strip()
