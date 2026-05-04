"""Dynamic quiz generation from retrieved academic context."""

from __future__ import annotations


class QuizModule:
    def generate(self, topic: str, context: str, graph_path: list[str], difficulty: str, count: int = 5) -> list[dict]:
        concepts = list(dict.fromkeys([*graph_path, topic.title()]))[:count]
        quiz = []
        for index, concept in enumerate(concepts):
            correct = f"It explains how {concept} supports {topic}"
            options = [
                correct,
                "It removes the need for prerequisite knowledge",
                "It is unrelated to the retrieved academic context",
                "It guarantees memorization without practice",
            ]
            shift = index % 4
            ordered = options[shift:] + options[:shift]
            quiz.append({
                "question": f"Why is {concept} important when learning {topic}?",
                "options": ordered,
                "correct_index": ordered.index(correct),
                "explanation": f"{concept} appears in the learning path or retrieved notes, so it helps structure the topic.",
                "difficulty": difficulty,
            })
        return quiz[:count]

