"""FastAPI adapter for the Academic AI Companion frontend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str = "default"


class QuizScoreRequest(BaseModel):
    user_id: str = "default"
    score: float = Field(..., ge=0, le=1)
    engagement_time: float = Field(default=30.0, ge=0)


@lru_cache(maxsize=1)
def get_system():
    from backend.core.agct_system import AGCTSystem

    return AGCTSystem()


app = FastAPI(
    title="Academic AI Companion API",
    description="Retrieval, Graph-CoT reasoning, verification, quiz, flashcard, and personalization API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> Dict[str, Any]:
    system = get_system()
    index_stats = system.retriever.get_index_stats()
    return {
        **index_stats,
        "graph_nodes": system.graph_engine.graph.number_of_nodes(),
        "graph_edges": system.graph_engine.graph.number_of_edges(),
    }


@app.get("/api/graph")
def graph() -> Dict[str, List[Dict[str, Any]]]:
    from backend.graph.cognitive_graph import CognitiveGraphEngine

    engine = CognitiveGraphEngine()
    engine.build_sample_graph()
    graph_engine = engine.graph
    nodes = [{"id": node, "label": node} for node in graph_engine.nodes()]
    edges = [
        {
            "source": source,
            "target": target,
            "relation": data.get("relation", "related"),
        }
        for source, target, data in graph_engine.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


@app.get("/api/progress/{user_id}")
def progress(user_id: str) -> Dict[str, Any]:
    from backend.agents.rl_personalization import RLPersonalizationAgent
    from backend.config import DB_PATH

    difficulty = RLPersonalizationAgent(db_path=DB_PATH).get_difficulty(user_id)
    return {
        "user_id": user_id,
        "difficulty": difficulty,
        "recent_topics": ["Calculus", "Chain Rule", "Backpropagation"],
        "progress": 68 if difficulty == "moderate" else 84 if difficulty == "advanced" else 42,
        "recommendations": [
            "Review prerequisite concepts before advanced prompts.",
            "Generate a quiz after each explanation.",
            "Use flashcards to reinforce graph-path concepts.",
        ],
    }


@app.post("/api/query")
def query(payload: QueryRequest) -> Dict[str, Any]:
    try:
        return get_system().process_query(payload.query, user_id=payload.user_id)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "query": payload.query,
            "mode": "error",
            "topic_query": "",
            "difficulty": "moderate",
            "graph_path": [],
            "explanation": (
                "The tutoring backend could not complete this request. "
                "Check that the ML dependencies and local Ollama model are available."
            ),
            "verified": False,
            "verification_score": 0.0,
            "sources": [],
            "retrieval": {},
            "flashcards": [],
            "quiz": [],
        }


@app.post("/api/quiz-score")
def quiz_score(payload: QuizScoreRequest) -> Dict[str, Any]:
    system = get_system()
    system.rl_agent.track_performance(
        payload.user_id,
        quiz_score=payload.score,
        engagement_time=payload.engagement_time,
    )
    return {
        "status": "success",
        "difficulty": system.rl_agent.get_difficulty(payload.user_id),
    }
