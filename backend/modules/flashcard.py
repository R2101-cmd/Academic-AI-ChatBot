"""Dynamic flashcard generation from retrieved context."""

from __future__ import annotations

import re


class FlashcardModule:
    def generate(self, topic: str, context: str, graph_path: list[str], count: int = 5) -> list[dict]:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", context) if len(item.strip()) > 40]
        concepts = list(dict.fromkeys([*graph_path, topic.title()]))[:count]
        cards = []
        for index, concept in enumerate(concepts):
            back = sentences[index % len(sentences)] if sentences else f"{concept} is a key idea for understanding {topic}."
            cards.append({
                "front": concept,
                "back": back[:260],
                "hint": f"Connect {concept.lower()} to {topic}.",
            })
        return cards[:count]

