"""Generator agent that produces Graph-CoT guided answers with Ollama."""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

import requests

from backend.config import DEFAULT_DIFFICULTY, LLM_API_URL, LLM_MODEL, REQUEST_TIMEOUT

MAX_RESPONSE_CHARS = 2600


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
                "top_p": 0.9,
                "repeat_penalty": 1.18,
                "repeat_last_n": 128,
                "num_predict": 420,
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

    def _deduplicate_graph_path(self, graph_path: List[str]) -> List[str]:
        return [item for item in dict.fromkeys(step.strip() for step in graph_path if step and step.strip())]

    def _sentence_key(self, sentence: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()

    def _deduplicate_sentences(self, text: str, max_repeats: int = 1) -> str:
        """Remove repeated sentences and recursive path-like lines."""
        seen = {}
        output = []
        for part in re.split(r"(?<!\d)(?<=[.!?])\s+|\n+", text.strip()):
            sentence = part.strip()
            if not sentence:
                continue
            key = self._sentence_key(sentence)
            if not key:
                continue
            if " -> " in sentence and len(set(sentence.split(" -> "))) < len(sentence.split(" -> ")):
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= max_repeats:
                output.append(sentence)
        return "\n\n".join(output)

    def _truncate_response(self, text: str, limit: int = MAX_RESPONSE_CHARS) -> str:
        if len(text) <= limit:
            return text
        trimmed = text[:limit].rsplit(".", 1)[0].strip()
        return f"{trimmed}." if trimmed else text[:limit].strip()

    def _clean_explanation_output(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^student\s*:.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"^user\s*:.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"^user question\s*:.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"^difficulty level\s*:.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
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
        cleaned = re.sub(r"^(rules|return only).*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"^(topic request|difficulty)\s*:.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        cleaned = self._deduplicate_sentences(cleaned)
        cleaned = self._truncate_response(cleaned)
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
        graph_path = self._deduplicate_graph_path(graph_path)
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

        return self._clean_explanation_output(
            f"{summary}{path_note} In short, this answer is based directly on the retrieved study notes."
        )

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
        steps = self._deduplicate_graph_path(list(graph_path)) or ["Foundations", "Core Idea", "Application"]
        graph_path_text = " -> ".join(steps)
        context_block = self._truncate_response(context.strip(), limit=1800) or "No supporting academic context was retrieved."

        return f"""You are an academic tutor for students.

Answer the user's question using only the retrieved study context.
Give a concise, structured explanation in an educational tone.

Question:
{query}

Difficulty level:
{difficulty}

Graph-CoT learning path:
{graph_path_text}

Retrieved context:
{context_block}

Rules:
1. Start with a direct answer.
2. Use 3 to 5 short numbered steps.
3. Explain prerequisite ideas before advanced ideas.
4. Include one simple example when useful.
5. Do not repeat sentences, headings, paths, or the question.
6. Do not print prompt labels such as "Question", "Retrieved context", or "Graph-CoT learning path".
7. If context is incomplete, say what is missing instead of inventing facts.
8. End with one concise recap sentence.
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

    def _valid_quiz_items(self, payload, topic: str, count: int) -> List[dict]:
        if not isinstance(payload, list):
            return []

        topic_terms = {term for term in re.findall(r"[a-zA-Z]{4,}", topic.lower())}
        weak_option_patterns = [
            "it is a prerequisite",
            "it is unrelated",
            "it replaces the need",
            "all of the above",
            "none of the above",
        ]
        seen_questions = set()
        seen_option_sets = set()
        clean_items = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            question = str(raw.get("question", "")).strip()
            options = [str(option).strip() for option in raw.get("options", []) if str(option).strip()]
            explanation = str(raw.get("explanation", "")).strip()
            correct_index = raw.get("correct_index", 0)
            try:
                correct_index = int(correct_index)
            except (TypeError, ValueError):
                correct_index = 0

            question_key = self._sentence_key(question)
            if not question or question_key in seen_questions:
                continue
            if len(options) != 4 or len(set(option.lower() for option in options)) != 4:
                continue
            option_key = tuple(sorted(option.lower() for option in options))
            if option_key in seen_option_sets:
                continue
            if any(pattern in option.lower() for option in options for pattern in weak_option_patterns):
                continue
            if not 0 <= correct_index < 4:
                continue
            if topic_terms and not any(term in (question + " " + explanation).lower() for term in topic_terms):
                continue

            seen_questions.add(question_key)
            seen_option_sets.add(option_key)
            clean_items.append(
                {
                    "question": question,
                    "options": options[:4],
                    "correct_index": correct_index,
                    "explanation": explanation or f"This checks a key idea from {topic}.",
                    "difficulty": raw.get("difficulty", "moderate"),
                }
            )
            if len(clean_items) >= count:
                break
        return clean_items

    def _valid_flashcards(self, payload, topic: str, count: int) -> List[dict]:
        if not isinstance(payload, list):
            return []

        seen_fronts = set()
        clean_cards = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            front = str(raw.get("front", "")).strip()
            back = self._truncate_response(str(raw.get("back", "")).strip(), limit=260)
            hint = str(raw.get("hint", "")).strip()
            key = self._sentence_key(front)
            if not front or not back or key in seen_fronts:
                continue
            seen_fronts.add(key)
            clean_cards.append(
                {
                    "front": front,
                    "back": back,
                    "hint": hint or f"Connect this term to {topic}.",
                }
            )
            if len(clean_cards) >= count:
                break
        return clean_cards

    def _topic_flashcard_templates(self, topic: str, graph_path: List[str], count: int) -> List[dict]:
        topic_lower = topic.lower()
        if "backprop" in topic_lower:
            cards = [
                ("Loss Function", "A mathematical measure of how far a model's prediction is from the correct answer."),
                ("Gradient", "A vector of derivatives that shows how each weight should change to reduce error."),
                ("Chain Rule", "A calculus rule used to compute gradients through nested or layered functions."),
                ("Weight Update", "The step where model parameters are adjusted using gradients and a learning rate."),
                ("Gradient Descent", "An optimization method that reduces loss by moving weights opposite the gradient."),
            ]
        elif "chain rule" in topic_lower:
            cards = [
                ("Composite Function", "A function built by applying one function inside another."),
                ("Outer Function", "The function differentiated after accounting for the inner function's change."),
                ("Inner Function", "The nested function whose derivative is multiplied in the chain rule."),
                ("Derivative", "A measure of how quickly a function changes with respect to its input."),
                ("Backpropagation", "A neural-network training method that repeatedly applies the chain rule."),
            ]
        else:
            concepts = list(dict.fromkeys([concept for concept in graph_path if concept] + [topic.title()]))
            base_cards = [
                (
                    topic.title(),
                    f"The main idea of {topic}: understand what it means, why it matters, and where it is used.",
                ),
                (
                    "Core Principle",
                    f"The central rule or relationship that explains how {topic} works in problems and examples.",
                ),
                (
                    "Worked Example",
                    f"A concrete use case that shows how to apply {topic} step by step instead of memorizing it.",
                ),
                (
                    "Common Mistake",
                    f"A frequent error in {topic} is skipping prerequisite ideas or applying a rule without checking conditions.",
                ),
                (
                    "Practice Strategy",
                    f"To study {topic}, explain the concept aloud, solve one example, then test yourself with a related question.",
                ),
            ]
            concept_cards = [
                (
                    concept,
                    f"{concept} supports {topic} by adding a prerequisite idea, method, or application you should review.",
                )
                for concept in concepts
            ]
            cards = list(dict.fromkeys(base_cards + concept_cards))

        return [
            {
                "front": front,
                "back": back,
                "hint": f"Connect {front.lower()} to {topic}.",
            }
            for front, back in cards[:count]
        ]

    def _topic_quiz_templates(self, topic: str, graph_path: List[str], difficulty: str, count: int) -> List[dict]:
        topic_lower = topic.lower()
        if "backprop" in topic_lower:
            items = [
                {
                    "question": "Why is the chain rule important in backpropagation?",
                    "options": [
                        "It computes gradients across layered functions",
                        "It initializes neural network weights randomly",
                        "It removes the need for a loss function",
                        "It converts labels into unsupervised data",
                    ],
                    "correct_index": 0,
                    "explanation": "Backpropagation applies the chain rule to pass gradient information through layers.",
                },
                {
                    "question": "What does a loss function provide during neural network training?",
                    "options": [
                        "A measure of prediction error to minimize",
                        "A fixed list of hidden layers",
                        "A replacement for the training dataset",
                        "A rule for choosing activation names",
                    ],
                    "correct_index": 0,
                    "explanation": "The loss function quantifies error so gradients can guide parameter updates.",
                },
                {
                    "question": "In a weight update, what is the role of the learning rate?",
                    "options": [
                        "It controls the size of each parameter adjustment",
                        "It stores all previous training examples",
                        "It decides the number of input features",
                        "It guarantees zero training error",
                    ],
                    "correct_index": 0,
                    "explanation": "The learning rate scales the gradient step during optimization.",
                },
                {
                    "question": "A model predicts poorly on one example. What does backpropagation compute first for learning?",
                    "options": [
                        "How the loss changes with respect to model parameters",
                        "A new dataset with different labels",
                        "The final accuracy of all future epochs",
                        "A diagram of the network architecture only",
                    ],
                    "correct_index": 0,
                    "explanation": "Backpropagation computes gradients that connect error to each trainable parameter.",
                },
                {
                    "question": "Which sequence best describes backpropagation?",
                    "options": [
                        "Compute loss, calculate gradients, update weights",
                        "Delete errors, freeze weights, skip optimization",
                        "Choose labels, remove layers, sort predictions",
                        "Normalize text, create tokens, render output",
                    ],
                    "correct_index": 0,
                    "explanation": "Training uses the loss to compute gradients and update weights.",
                },
            ]
        elif "chain rule" in topic_lower:
            items = [
                {
                    "question": "What type of function is the chain rule designed to differentiate?",
                    "options": [
                        "A composite function",
                        "A constant table",
                        "An unordered dataset",
                        "A linear equation only",
                    ],
                    "correct_index": 0,
                    "explanation": "The chain rule handles functions nested inside other functions.",
                },
                {
                    "question": "If y = f(g(x)), what must be multiplied when applying the chain rule?",
                    "options": [
                        "The derivative of f with respect to g and the derivative of g with respect to x",
                        "The values of x and y without differentiation",
                        "Only the final output value",
                        "The number of terms in the expression",
                    ],
                    "correct_index": 0,
                    "explanation": "The outer derivative is multiplied by the inner derivative.",
                },
                {
                    "question": "Why does the chain rule appear in neural networks?",
                    "options": [
                        "Layers form nested functions from input to loss",
                        "Weights are always integers",
                        "Training avoids derivatives",
                        "Outputs never depend on earlier layers",
                    ],
                    "correct_index": 0,
                    "explanation": "Neural networks compose transformations, so gradients require the chain rule.",
                },
                {
                    "question": "What is a common mistake when using the chain rule?",
                    "options": [
                        "Forgetting to multiply by the inner derivative",
                        "Using examples to understand notation",
                        "Identifying the outer function first",
                        "Checking units or variables",
                    ],
                    "correct_index": 0,
                    "explanation": "The inner derivative is essential because the inner function also changes.",
                },
                {
                    "question": "Which statement best explains the chain rule conceptually?",
                    "options": [
                        "It tracks how a change passes through dependent functions",
                        "It states that every derivative is zero",
                        "It removes variables from equations",
                        "It only applies to probability questions",
                    ],
                    "correct_index": 0,
                    "explanation": "The chain rule describes how changes propagate through a composition.",
                },
            ]
        else:
            concepts = list(dict.fromkeys([concept for concept in graph_path if concept] + [topic.title(), "Application"]))
            primary = concepts[0]
            secondary = concepts[1] if len(concepts) > 1 else "the prerequisite concept"
            items = [
                {
                    "question": f"When studying {topic}, what is the best first step before solving advanced problems?",
                    "options": [
                        f"Clarify the meaning and role of {primary}",
                        "Memorize answers without checking the method",
                        "Skip examples and start with the hardest case",
                        "Change the topic whenever notation appears",
                    ],
                    "correct_index": 0,
                    "explanation": f"{primary} gives a foundation for understanding {topic}.",
                },
                {
                    "question": f"A student understands definitions in {topic} but struggles with application. What should they do next?",
                    "options": [
                        "Ignore the definitions and guess from keywords",
                        f"Work through one example that connects {secondary} to the main idea",
                        "Only reread the title of the notes",
                        "Assume every problem uses the same final answer",
                    ],
                    "correct_index": 1,
                    "explanation": "Application improves when the learner connects a prerequisite idea to a worked example.",
                },
                {
                    "question": f"Which question best checks real understanding of {topic}?",
                    "options": [
                        "Can I copy the same sentence from the notes?",
                        "Can I avoid explaining the concept in my own words?",
                        "Can I explain why the method works in a new example?",
                        "Can I list terms without knowing their relationships?",
                    ],
                    "correct_index": 2,
                    "explanation": "Transfer to a new example is a stronger signal than memorized wording.",
                },
                {
                    "question": f"If an answer about {topic} feels confusing, what is the most useful learning move?",
                    "options": [
                        "Move directly to a harder topic",
                        "Remove all prerequisite concepts",
                        "Choose an answer by option length",
                        f"Ask for a simpler explanation with a concrete {topic} example",
                    ],
                    "correct_index": 3,
                    "explanation": "A simpler example can reveal the missing step in the concept progression.",
                },
                {
                    "question": f"Why are prerequisite concepts useful when learning {topic}?",
                    "options": [
                        "They show which earlier ideas support the current concept",
                        "They make practice unnecessary",
                        "They guarantee every answer is numerical",
                        "They replace the need for feedback",
                    ],
                    "correct_index": 0,
                    "explanation": "Prerequisites make the learning path easier to follow and diagnose.",
                },
            ]

        for index, item in enumerate(items):
            target_index = index % 4
            correct_option = item["options"][item["correct_index"]]
            options = [option for idx, option in enumerate(item["options"]) if idx != item["correct_index"]]
            options.insert(target_index, correct_option)
            item["options"] = options
            item["correct_index"] = target_index
            item["difficulty"] = difficulty
        return items[:count]

    def _merge_unique_quiz(self, primary: List[dict], fallback: List[dict], count: int) -> List[dict]:
        merged = []
        seen_questions = set()
        seen_option_sets = set()
        for item in [*primary, *fallback]:
            question_key = self._sentence_key(item.get("question", ""))
            option_key = tuple(sorted(str(option).lower() for option in item.get("options", [])))
            if not question_key or question_key in seen_questions or option_key in seen_option_sets:
                continue
            seen_questions.add(question_key)
            seen_option_sets.add(option_key)
            merged.append(item)
            if len(merged) >= count:
                break
        return merged

    def _merge_unique_flashcards(self, primary: List[dict], fallback: List[dict], count: int) -> List[dict]:
        merged = []
        seen_fronts = set()
        for card in [*primary, *fallback]:
            front_key = self._sentence_key(card.get("front", ""))
            if not front_key or front_key in seen_fronts:
                continue
            seen_fronts.add(front_key)
            merged.append(card)
            if len(merged) >= count:
                break
        return merged

    def generate_flashcards(
        self,
        query: str,
        context: str,
        graph_path: List[str],
        difficulty: str = DEFAULT_DIFFICULTY,
        count: int = 5,
    ) -> List[dict]:
        """Generate study flashcards only when explicitly requested."""
        graph_path = self._deduplicate_graph_path(graph_path)
        context = self._truncate_response(context, limit=1600)
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
Use only topic-specific concepts from the retrieved context and learning path.
Avoid duplicate cards, generic AI filler, and unrelated subjects.
"""
        result = self._call_model(prompt, temperature=0.3, seed=101)
        if result.lower().startswith("error:"):
            return self._topic_flashcard_templates(query, graph_path, count)

        try:
            payload = self._extract_json_payload(result)
            clean_cards = self._valid_flashcards(payload, query, count)
            fallback_cards = self._topic_flashcard_templates(query, graph_path, count)
            merged_cards = self._merge_unique_flashcards(clean_cards, fallback_cards, count)
            if len(merged_cards) >= count:
                return merged_cards
        except Exception:
            pass

        return self._topic_flashcard_templates(query, graph_path, count)

    def generate_quiz(
        self,
        query: str,
        context: str,
        graph_path: List[str],
        difficulty: str = DEFAULT_DIFFICULTY,
        count: int = 5,
    ) -> List[dict]:
        """Generate MCQ practice with answer reveal support."""
        graph_path = self._deduplicate_graph_path(graph_path)
        context = self._truncate_response(context, limit=1600)
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
- "difficulty": "basic", "moderate", or "advanced"

Create topic-specific conceptual and application questions.
Use academically meaningful distractors that are plausible but wrong.
Vary the correct answer position across questions.
Do not repeat questions, option sets, generic distractors, or unrelated subjects.
"""
        result = self._call_model(prompt, temperature=0.3, seed=202)
        if result.lower().startswith("error:"):
            return self._topic_quiz_templates(query, graph_path, difficulty, count)

        try:
            payload = self._extract_json_payload(result)
            clean_quiz = self._valid_quiz_items(payload, query, count)
            fallback_quiz = self._topic_quiz_templates(query, graph_path, difficulty, count)
            merged_quiz = self._merge_unique_quiz(clean_quiz, fallback_quiz, count)
            if len(merged_quiz) >= count:
                return merged_quiz
        except Exception:
            pass

        return self._topic_quiz_templates(query, graph_path, difficulty, count)
