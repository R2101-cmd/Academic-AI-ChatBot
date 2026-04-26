"""Main orchestration layer for the academic Graph-CoT tutor."""

from __future__ import annotations

from typing import Dict

from backend.agents.reasoner_agent import ReasonerAgent
from backend.agents.retriever_agent import RetrieverAgent
from backend.agents.rl_personalization import RLPersonalizationAgent
from backend.agents.verifier_agent import VerifierAgent
from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_PATH, DB_PATH, FAISS_K
from backend.graph.cognitive_graph import CognitiveGraphEngine
from backend.rag.rag_setup import RAGPipeline
from backend.utils.validators import detect_request_mode, extract_topic_query


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

    def process_query(self, query: str, user_id: str = "default") -> Dict:
        """Run the end-to-end pipeline for a student query."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return {"status": "rejected", "error": "Empty query"}

        mode = detect_request_mode(cleaned_query)
        topic_query = extract_topic_query(cleaned_query)
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

        return {
            "status": "success",
            "query": cleaned_query,
            "mode": mode,
            "topic_query": topic_query,
            "difficulty": difficulty,
            "graph_path": retrieval_result["graph_path"],
            "explanation": explanation,
            "verified": is_verified,
            "verification_score": similarity,
            "sources": retrieval_result["combined_docs"],
            "retrieval": retrieval_result,
            "flashcards": flashcards,
            "quiz": quiz,
        }
