"""End-to-end Academic AI Companion pipeline."""

from __future__ import annotations

from functools import cached_property

from backend.config import FAISS_K
from backend.modules.flashcard import FlashcardModule
from backend.modules.generation import GenerationModule
from backend.modules.graph_cot import GraphCoTModule
from backend.modules.nlp import NLPProcessor
from backend.modules.personalization import PersonalizationModule
from backend.modules.quiz import QuizModule
from backend.modules.retrieval import RetrievalModule
from backend.modules.verification import VerificationModule


class AcademicAICompanion:
    """System flow: NLP -> embeddings -> FAISS -> Graph-CoT -> LLM -> verification -> study assets."""

    def __init__(self) -> None:
        self.nlp = NLPProcessor()
        self.retrieval = RetrievalModule()
        self.graph_cot = GraphCoTModule()
        self.generation = GenerationModule()
        self.quiz = QuizModule()
        self.flashcard = FlashcardModule()
        self.verification = VerificationModule()
        self.personalization = PersonalizationModule()

    def process_query(self, query: str, user_id: str = "default") -> dict:
        processed = self.nlp.preprocess(query)
        if not processed.is_academic:
            return {
                "status": "rejected",
                "error": processed.rejection_reason,
                "query": processed.raw,
                "mode": processed.mode,
                "topic_query": processed.topic,
                "difficulty": self.personalization.difficulty(user_id),
                "graph_path": [],
                "explanation": "I can help with academic topics only. Try asking about a subject, concept, formula, programming topic, or exam preparation.",
                "verified": False,
                "verification_score": 0.0,
                "verification_status": "Rejected by academic-domain validator",
                "sources": [],
                "retrieval": {},
                "retrieval_notes": [],
                "quiz": [],
                "flashcards": [],
                "suggested_questions": ["Explain machine learning", "Create quiz on calculus", "Generate flashcards for algorithms"],
            }

        difficulty = self.personalization.difficulty(user_id)
        retrieval = self.retrieval.retrieve(processed.topic, k=FAISS_K)
        concepts = [concept for result in retrieval["results"] for concept in result["concepts"]]
        graph_path = self.graph_cot.build_path(processed.topic, concepts)
        explanation = self.generation.generate(processed.cleaned, retrieval["combined_context"], graph_path, difficulty)
        verification = self.verification.verify(processed.topic, explanation, retrieval["combined_context"], graph_path)
        self.personalization.record(user_id, processed.topic, verification["score"])

        quiz = self.quiz.generate(processed.topic, retrieval["combined_context"], graph_path, difficulty) if processed.mode == "quiz" else []
        flashcards = self.flashcard.generate(processed.topic, retrieval["combined_context"], graph_path) if processed.mode == "flashcards" else []

        return {
            "status": "success",
            "query": processed.raw,
            "mode": processed.mode,
            "topic_query": processed.topic,
            "difficulty": difficulty,
            "graph_path": graph_path,
            "explanation": explanation,
            "verified": verification["verified"],
            "verification_score": verification["score"],
            "verification_status": verification["status"],
            "sources": [item["text"] for item in retrieval["results"]],
            "retrieval": retrieval,
            "retrieval_notes": [
                {
                    "title": item["title"],
                    "preview": item["text"][:220],
                    "full_text": item["text"],
                    "relevance_score": item["score"],
                    "concepts": item["concepts"],
                }
                for item in retrieval["results"]
            ],
            "quiz": quiz,
            "flashcards": flashcards,
            "suggested_questions": [
                f"Create quiz on {processed.topic}",
                f"Generate flashcards for {processed.topic}",
                f"Show a simpler example of {processed.topic}",
            ],
        }

    def stats(self) -> dict:
        return {
            **self.retrieval.stats(),
            "graph_nodes": self.graph_cot.graph.number_of_nodes(),
            "graph_edges": self.graph_cot.graph.number_of_edges(),
        }

