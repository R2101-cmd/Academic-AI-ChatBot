"""Verification checks for retrieved context and generated explanations."""

from __future__ import annotations

import re


class VerificationModule:
    def verify(self, topic: str, explanation: str, context: str, graph_path: list[str]) -> dict:
        topic_terms = set(re.findall(r"[a-zA-Z]{4,}", topic.lower()))
        explanation_terms = set(re.findall(r"[a-zA-Z]{4,}", explanation.lower()))
        context_terms = set(re.findall(r"[a-zA-Z]{4,}", context.lower()))

        topic_overlap = len(topic_terms & explanation_terms) / max(1, len(topic_terms))
        grounding = len(explanation_terms & context_terms) / max(1, min(len(explanation_terms), 30))
        path_hits = sum(1 for node in graph_path if node.lower() in explanation.lower()) / max(1, len(graph_path))
        score = round((topic_overlap * 0.35) + (grounding * 0.45) + (path_hits * 0.2), 2)

        return {
            "verified": score >= 0.45,
            "score": score,
            "status": "Verified against retrieved academic context" if score >= 0.45 else "Needs more supporting context",
        }

