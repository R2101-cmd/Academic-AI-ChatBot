"""FastAPI adapter for the Academic AI Companion frontend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

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
    from backend.modules.orchestrator import AcademicAICompanion

    return AcademicAICompanion()


app = FastAPI(
    title="Academic AI Companion API",
    description="Academic RAG, Graph-CoT, TinyLlama generation, verification, quiz, flashcard, and personalization API.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return get_system().stats()


@app.get("/api/graph")
def graph() -> dict[str, list[dict[str, str]]]:
    return get_system().graph_cot.graph_payload()


@app.get("/api/progress/{user_id}")
def progress(user_id: str) -> dict[str, Any]:
    return get_system().personalization.progress(user_id)


@app.post("/api/query")
def query(payload: QueryRequest) -> dict[str, Any]:
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
            "explanation": "The backend could not complete this request. Check dependencies and Ollama/TinyLlama setup.",
            "verified": False,
            "verification_score": 0.0,
            "verification_status": "Backend error",
            "sources": [],
            "retrieval": {},
            "retrieval_notes": [],
            "flashcards": [],
            "quiz": [],
            "suggested_questions": [],
        }


@app.post("/api/quiz-score")
def quiz_score(payload: QuizScoreRequest) -> dict[str, Any]:
    system = get_system()
    system.personalization.record(payload.user_id, "quiz practice", payload.score)
    return {
        "status": "success",
        "difficulty": system.personalization.difficulty(payload.user_id),
    }

