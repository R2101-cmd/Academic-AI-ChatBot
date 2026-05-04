"""SQLite-backed personalization module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.config import DB_PATH


class PersonalizationModule:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def record(self, user_id: str, topic: str, score: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO learner_activity (user_id, topic, score) VALUES (?, ?, ?)",
                (user_id, topic, score),
            )

    def difficulty(self, user_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT score FROM learner_activity WHERE user_id = ? ORDER BY id DESC LIMIT 5",
                (user_id,),
            ).fetchall()
        if not rows:
            return "moderate"
        average = sum(row[0] for row in rows) / len(rows)
        if average < 0.6:
            return "basic"
        if average > 0.82:
            return "advanced"
        return "moderate"

    def progress(self, user_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT topic, score FROM learner_activity WHERE user_id = ? ORDER BY id DESC LIMIT 8",
                (user_id,),
            ).fetchall()
        topics = list(dict.fromkeys(row[0].title() for row in rows))[:5]
        average = round((sum(row[1] for row in rows) / len(rows)) * 100) if rows else 68
        return {
            "user_id": user_id,
            "difficulty": self.difficulty(user_id),
            "recent_topics": topics or ["Calculus", "Chain Rule", "Backpropagation"],
            "progress": average,
            "recommendations": [
                "Review the first concept in your Graph-CoT path.",
                "Generate a quiz after every explanation.",
                "Convert weak concepts into flashcards.",
            ],
        }

