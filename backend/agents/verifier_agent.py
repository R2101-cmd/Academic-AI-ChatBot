"""Verifier agent for self-consistency and answer quality checks."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import DEFAULT_VERIFICATION_THRESHOLD


class VerifierAgent:
    """Verify explanations using semantic agreement and basic sanity rules."""

    def __init__(
        self,
        reasoner,
        model: SentenceTransformer,
        threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
    ) -> None:
        self.reasoner = reasoner
        self.model = model
        self.threshold = threshold

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        embeddings = self.model.encode([text_a, text_b], convert_to_numpy=True)
        emb_a, emb_b = embeddings[0], embeddings[1]
        return float((emb_a @ emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))

    def _contains_required_path(self, explanation: str, graph_path: List[str]) -> bool:
        explanation_lower = explanation.lower()
        required = [
            concept.lower()
            for concept in graph_path[: min(3, len(graph_path))]
            if len(concept.strip()) > 2
        ]
        if not required:
            return True
        return all(concept in explanation_lower for concept in required)

    def verify(
        self,
        query: str,
        context: str,
        graph_path: List[str],
        difficulty: str,
        samples: int = 2,
    ) -> Tuple[str, bool, float]:
        """Generate multiple answers and verify by semantic agreement."""
        candidates: List[str] = []
        similarities: List[float] = []

        total_samples = max(1, samples)

        for idx in range(total_samples):
            candidates.append(
                self.reasoner.generate_explanation(
                    query=query,
                    context=context,
                    graph_path=graph_path,
                    difficulty=difficulty,
                    temperature=0.2 + (idx * 0.15),
                    seed=42 + idx,
                )
            )

        for idx in range(len(candidates) - 1):
            similarities.append(self._cosine_similarity(candidates[idx], candidates[idx + 1]))

        primary_answer = candidates[0]
        verification_score = (
            float(sum(similarities) / len(similarities))
            if similarities
            else (1.0 if not primary_answer.lower().startswith("error:") else 0.0)
        )
        is_verified = (
            verification_score >= self.threshold
            and not primary_answer.lower().startswith("error:")
            and self._contains_required_path(primary_answer, graph_path)
        )
        return primary_answer, is_verified, verification_score

    def verify_with_details(
        self,
        query: str,
        context: str,
        graph_path: List[str],
        difficulty: str,
    ) -> Dict:
        """Return a richer verification payload for debugging or UI use."""
        answer, verified, score = self.verify(query, context, graph_path, difficulty)
        return {
            "answer": answer,
            "verified": verified,
            "score": score,
            "threshold": self.threshold,
            "graph_path": graph_path,
        }
