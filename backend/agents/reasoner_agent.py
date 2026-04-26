"""Generator agent that produces Graph-CoT guided answers with Ollama."""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

import requests

from backend.config import DEFAULT_DIFFICULTY, LLM_API_URL, LLM_MODEL, REQUEST_TIMEOUT


class ReasonerAgent:
    """Generate grounded academic explanations and study materials."""

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        api_url: str = LLM_API_URL,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.model = model_name
        self.api_url = api_url
        self.timeout = timeout

    def _call_model(
        self,
        prompt: str,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            return body.get("response", "").strip() or "No response was generated."
        except Exception as exc:
            return (
                f"Error: {exc}\n"
                f"Tip: make sure Ollama is running and the model '{self.model}' responds locally. "
                f"If it is too slow, try a smaller model or increase REQUEST_TIMEOUT."
            )

    def _clean_explanation_output(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^student\s*:.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"^user\s*:.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(
            r"^graph-cot learning path\s*:.*?$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        cleaned = re.sub(
            r"^retrieved context\s*:.*?$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned or text.strip()

    def _looks_like_prompt_echo(self, text: str) -> bool:
        lowered = text.lower()
        echo_markers = [
            "user question:",
            "retrieved context:",
            "return only the explanation text",
            "do not repeat the user's question",
            "graph-cot learning path:",
        ]
        return sum(marker in lowered for marker in echo_markers) >= 2

    def _fallback_explanation(
        self,
        query: str,
        context: str,
        graph_path: List[str],
    ) -> str:
        """Build a short grounded explanation directly from retrieved context."""
        lines = []
        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "`" in line:
                continue
            if line.lower().startswith("available topic collections"):
                continue
            if line.lower().startswith("this folder contains multiple topic files"):
                continue
            if line.lower().startswith("the assistant should answer using the most relevant study content"):
                continue
            lines.append(line)

        text = " ".join(lines)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        useful_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 30]
        summary = " ".join(useful_sentences[:3]).strip()

        if not summary:
            summary = "I could not find enough retrieved context to answer this clearly yet."

        path_note = ""
        relevant_path = [concept for concept in graph_path if concept not in {"Foundations", "Core Concept", "Application"}]
        if relevant_path:
            path_note = f" Related concepts include {' -> '.join(relevant_path[:3])}."

        return f"{summary}{path_note} In short, this answer is based directly on the retrieved study notes."

    def _extract_json_payload(self, text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("Model did not return valid JSON.")

    def build_prompt(
        self,
        query: str,
        context: str,
        graph_path: Iterable[str],
        difficulty: str = DEFAULT_DIFFICULTY,
    ) -> str:
        """Create a clean Graph-CoT prompt for the local model."""
        steps = list(graph_path) or ["Foundations", "Core Idea", "Application"]
        graph_path_text = " -> ".join(steps)
        context_block = context.strip() or "No supporting academic context was retrieved."

        return f"""You are a study assistant.

Answer the user's question using only the retrieved study context.
Follow the Graph-CoT learning path when it is relevant and explain in a simple, student-friendly way.

User question:
{query}

Difficulty level:
{difficulty}

Graph-CoT learning path:
{graph_path_text}

Retrieved context:
{context_block}

Rules:
1. Do not repeat the user's question.
2. Do not start lines with "User:".
3. Do not print the label "Graph-CoT learning path:".
4. Start with a short direct answer.
5. Then explain step by step, using the graph path order if it fits the topic.
6. Mention relevant concepts from the path clearly when they apply.
7. If the context is incomplete, say what is missing instead of inventing facts.
8. End with a one-line recap.
Return only the explanation text.
"""

    def generate_explanation(
        self,
        query: str,
        context: str,
        graph_path: List[str],
        difficulty: str = DEFAULT_DIFFICULTY,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ) -> str:
        """Call Ollama and generate a grounded explanation."""
        prompt = self.build_prompt(query, context, graph_path, difficulty)
        result = self._call_model(prompt, temperature=temperature, seed=seed)
        if result.lower().startswith("error:"):
            return self._fallback_explanation(query, context, graph_path)

        cleaned = self._clean_explanation_output(result)
        if self._looks_like_prompt_echo(cleaned):
            return self._fallback_explanation(query, context, graph_path)
        return cleaned

    def generate_flashcards(
        self,
        query: str,
        context: str,
        graph_path: List[str],
        difficulty: str = DEFAULT_DIFFICULTY,
        count: int = 5,
    ) -> List[dict]:
        """Generate study flashcards only when explicitly requested."""
        prompt = f"""You are creating study flashcards for a student.

Topic request:
{query}

Difficulty:
{difficulty}

Graph-CoT learning path:
{" -> ".join(graph_path)}

Retrieved context:
{context}

Return ONLY valid JSON as an array of {count} objects.
Each object must have:
- "front": short question or cue
- "back": concise answer
- "hint": one short memory aid
Keep the cards easy to revise and student-friendly.
"""
        result = self._call_model(prompt, temperature=0.3, seed=101)
        if result.lower().startswith("error:"):
            return [{"front": "Generation error", "back": result, "hint": "Check Ollama."}]

        try:
            payload = self._extract_json_payload(result)
            if isinstance(payload, list):
                return payload
        except Exception:
            pass

        fallback_cards = []
        for concept in graph_path[:count]:
            fallback_cards.append(
                {
                    "front": f"What is {concept}?",
                    "back": f"{concept} is an important part of this topic's learning path.",
                    "hint": f"Connect {concept} to the next step in the path.",
                }
            )
        return fallback_cards

    def generate_quiz(
        self,
        query: str,
        context: str,
        graph_path: List[str],
        difficulty: str = DEFAULT_DIFFICULTY,
        count: int = 3,
    ) -> List[dict]:
        """Generate MCQ practice with answer reveal support."""
        prompt = f"""You are creating a multiple-choice quiz for a student.

Topic request:
{query}

Difficulty:
{difficulty}

Graph-CoT learning path:
{" -> ".join(graph_path)}

Retrieved context:
{context}

Return ONLY valid JSON as an array of {count} objects.
Each object must have:
- "question": string
- "options": array of exactly 4 short options
- "correct_index": integer from 0 to 3
- "explanation": short explanation

Make the quiz easy to study from, not tricky.
"""
        result = self._call_model(prompt, temperature=0.3, seed=202)
        if result.lower().startswith("error:"):
            return [
                {
                    "question": "Quiz generation error",
                    "options": ["Check Ollama", "Retry later", "Use smaller model", "All of the above"],
                    "correct_index": 3,
                    "explanation": result,
                }
            ]

        try:
            payload = self._extract_json_payload(result)
            if isinstance(payload, list):
                return payload
        except Exception:
            pass

        fallback_quiz = []
        for concept in graph_path[:count]:
            fallback_quiz.append(
                {
                    "question": f"Which concept is part of the learning path for {query}?",
                    "options": [concept, "Random topic", "Unrelated concept", "None of these"],
                    "correct_index": 0,
                    "explanation": f"{concept} appears in the Graph-CoT learning path.",
                }
            )
        return fallback_quiz
