"""Main orchestration layer for the academic Graph-CoT tutor."""

from __future__ import annotations

import re
from typing import Dict

from backend.agents.reasoner_agent import ReasonerAgent
from backend.agents.retriever_agent import RetrieverAgent
from backend.agents.rl_personalization import RLPersonalizationAgent
from backend.agents.verifier_agent import VerifierAgent
from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_PATH, DB_PATH, FAISS_K
from backend.graph.cognitive_graph import CognitiveGraphEngine
from backend.rag.rag_setup import RAGPipeline
from backend.session_manager import SessionManager
from backend.utils.validators import detect_request_mode, extract_topic_query, is_followup_command


class AGCTSystem:
    """Full Member 2 pipeline: retrieval, generation, verification, Graph-CoT."""

    def __init__(self, sample_data_path: str = DATA_PATH) -> None:
        self.rag_pipeline = RAGPipeline(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.rag_pipeline.setup(sample_data_path)

        self.documents = self.rag_pipeline.documents
        self.model = self.rag_pipeline.model
        self.faiss_index = self.rag_pipeline.index

        self.graph_engine = CognitiveGraphEngine()
        self.graph_engine.build_sample_graph()

        self.retriever = RetrieverAgent(
            self.faiss_index,
            self.documents,
            self.model,
            self.graph_engine,
        )
        self.reasoner = ReasonerAgent()
        self.verifier = VerifierAgent(self.reasoner, self.model)
        self.rl_agent = RLPersonalizationAgent(db_path=DB_PATH)
        self.sessions = SessionManager()

    def _resolve_topic(self, cleaned_query: str, user_id: str) -> tuple[str, bool]:
        """Use session memory when the student sends a command-only follow-up."""
        session = self.sessions.get(user_id)
        extracted_topic = extract_topic_query(cleaned_query)
        followup = is_followup_command(cleaned_query)
        weak_topics = {"quiz", "mcq", "mcqs", "test", "flashcard", "flashcards", "revision", "review", "summary"}

        if (followup or extracted_topic.lower() in weak_topics) and session.latest_topic:
            return session.latest_topic, True

        return extracted_topic, False

    def _clean_preview(self, text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[a-z]{0,4}\s+", "", text)
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        complete = [sentence for sentence in sentences if sentence[-1:] in ".!?" and len(sentence) > 35]
        preview = " ".join((complete or sentences)[:max_sentences]).strip()
        if not preview:
            preview = text[:max_chars].strip()
        if len(preview) > max_chars:
            preview = preview[:max_chars].rsplit(" ", 1)[0].strip() + "."
        return preview

    def _retrieval_title(self, doc: str, topic: str, graph_path: list[str]) -> str:
        heading = re.search(r"#{1,3}\s+(.+)", doc)
        if heading:
            return heading.group(1).strip()
        lowered = doc.lower()
        for concept in reversed(graph_path):
            if concept.lower() in lowered:
                return concept
        return topic.title() if topic else "Retrieved Study Note"

    def _format_retrieval_notes(self, retrieval_result: Dict, topic: str) -> list[Dict]:
        docs = retrieval_result.get("combined_docs", [])
        semantic_docs = retrieval_result.get("semantic_docs", [])
        scores = retrieval_result.get("semantic_scores", [])
        graph_path = retrieval_result.get("graph_path", [])
        notes = []

        for index, doc in enumerate(docs[:5]):
            score = None
            if doc in semantic_docs:
                semantic_index = semantic_docs.index(doc)
                if semantic_index < len(scores):
                    score = round(float(scores[semantic_index]), 2)

            concepts = [
                concept
                for concept in graph_path
                if concept.lower() in doc.lower()
            ][:4]

            notes.append(
                {
                    "title": self._retrieval_title(doc, topic, graph_path),
                    "preview": self._clean_preview(doc),
                    "full_text": re.sub(r"\s+", " ", doc).strip(),
                    "relevance_score": score,
                    "concepts": concepts,
                }
            )

        return notes

    def _suggest_followups(self, topic: str, mode: str, graph_path: list[str]) -> list[str]:
        path_topics = [concept for concept in graph_path if concept.lower() != topic.lower()]
        suggestions = []
        if mode != "quiz":
            suggestions.append("Create quiz")
        if mode != "flashcards":
            suggestions.append("Generate flashcards")
        suggestions.append("Give revision")
        if path_topics:
            suggestions.append(f"Explain {path_topics[-1]}")
        suggestions.append("Show real-world example")
        return list(dict.fromkeys(suggestions))[:5]

    def process_query(self, query: str, user_id: str = "default") -> Dict:
        """Run the end-to-end pipeline for a student query."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return {"status": "rejected", "error": "Empty query"}

        self.sessions.add_turn(user_id, "user", cleaned_query)
        mode = detect_request_mode(cleaned_query)
        session = self.sessions.get(user_id)
        if is_followup_command(cleaned_query) and not session.latest_topic:
            message = (
                "Please mention a learning topic first. For example: "
                "'Explain backpropagation' or 'Create a quiz on the chain rule'."
            )
            return {
                "status": "success",
                "query": cleaned_query,
                "mode": mode,
                "topic_query": "",
                "used_memory_topic": False,
                "latest_topic": None,
                "difficulty": self.rl_agent.get_difficulty(user_id),
                "graph_path": [],
                "explanation": message,
                "verified": False,
                "verification_score": 0.0,
                "sources": [],
                "retrieval": {},
                "flashcards": [],
                "quiz": [],
            }

        topic_query, used_memory_topic = self._resolve_topic(cleaned_query, user_id)
        difficulty = self.rl_agent.get_difficulty(user_id)
        retrieval_result = self.retriever.retrieve(topic_query, k=FAISS_K)
        explanation, is_verified, similarity = self.verifier.verify(
            topic_query,
            retrieval_result["combined_context"],
            retrieval_result["graph_path"],
            difficulty,
            samples=1,
        )

        self.rl_agent.track_performance(
            user_id,
            quiz_score=0.75 if is_verified else 0.6,
            engagement_time=15.0,
        )

        flashcards = []
        quiz = []
        if mode == "flashcards":
            flashcards = self.reasoner.generate_flashcards(
                topic_query,
                retrieval_result["combined_context"],
                retrieval_result["graph_path"],
                difficulty,
            )
        elif mode == "quiz":
            quiz = self.reasoner.generate_quiz(
                topic_query,
                retrieval_result["combined_context"],
                retrieval_result["graph_path"],
                difficulty,
            )

        if not used_memory_topic:
            self.sessions.update_topic(user_id, topic_query, retrieval_result["graph_path"])
        self.sessions.add_turn(user_id, "assistant", explanation)
        self.sessions.complete_exchange(user_id, cleaned_query, explanation, topic_query)
        session = self.sessions.get(user_id)
        retrieval_notes = self._format_retrieval_notes(retrieval_result, topic_query)

        return {
            "status": "success",
            "query": cleaned_query,
            "mode": mode,
            "topic_query": topic_query,
            "used_memory_topic": used_memory_topic,
            "latest_topic": session.latest_topic,
            "difficulty": difficulty,
            "graph_path": retrieval_result["graph_path"],
            "explanation": explanation,
            "verified": is_verified,
            "verification_score": similarity,
            "sources": retrieval_result["combined_docs"],
            "retrieval": retrieval_result,
            "retrieval_notes": retrieval_notes,
            "suggested_questions": self._suggest_followups(topic_query, mode, retrieval_result["graph_path"]),
            "chat_history": session.chat_history,
            "flashcards": flashcards,
            "quiz": quiz,
        }
