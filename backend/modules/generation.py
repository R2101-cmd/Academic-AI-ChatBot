"""TinyLlama/Ollama generation with grounded fallback output."""

from __future__ import annotations

import re
from typing import Any

import requests

from backend.config import LLM_API_URL, LLM_MODEL, REQUEST_TIMEOUT


class GenerationModule:
    """Generate step-by-step academic explanations from retrieved context."""

    def __init__(self, model: str = LLM_MODEL, api_url: str = LLM_API_URL) -> None:
        self.model = model
        self.api_url = api_url

    def generate(self, query: str, context: str, graph_path: list[str], difficulty: str) -> str:
        prompt = self._prompt(query, context, graph_path, difficulty)
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 520},
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if text:
                path = " -> ".join(graph_path)
                return f"Learning path: {path}\n\n{self._clean(text)}"
        except Exception:
            pass
        return self._fallback(query, context, graph_path)

    def _prompt(self, query: str, context: str, graph_path: list[str], difficulty: str) -> str:
        return f"""You are an Academic AI Companion.
Use only the retrieved academic context.

Student query: {query}
Difficulty: {difficulty}
Graph-CoT learning path: {" -> ".join(graph_path)}
Retrieved context:
{context[:2200]}

Write a clear academic answer:
1. Direct answer first.
2. Then 3-5 numbered learning steps.
3. Mention the learning path naturally.
4. Include one short example if useful.
5. End with a recap.
Do not answer non-academic questions.
"""

    def _clean(self, text: str) -> str:
        text = re.sub(r"(?i)^(student query|retrieved context|difficulty|graph-cot learning path|learning path):.*$", "", text, flags=re.MULTILINE)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _fallback(self, query: str, context: str, graph_path: list[str]) -> str:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", context) if len(item.strip()) > 35]
        summary = " ".join(sentences[:3]) or "The retrieved notes do not contain enough detail for a complete answer."
        path = " -> ".join(graph_path)
        return (
            f"{summary}\n\n"
            f"1. Start with the foundation: {graph_path[0] if graph_path else 'the prerequisite idea'}.\n"
            f"2. Connect it to the main topic: {query}.\n"
            f"3. Practice by explaining the concept in your own words and solving one example.\n\n"
            f"Graph-CoT learning path: {path}\n\n"
            "Recap: this explanation is grounded in the retrieved academic notes."
        )
