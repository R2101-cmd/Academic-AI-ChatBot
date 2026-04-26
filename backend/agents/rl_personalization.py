"""RL personalization agent backed by SQLite learner-state storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class RLPersonalizationAgent:
    """Track user performance and recommend difficulty."""

    def __init__(self, db_path: str = "backend/data/session.db") -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_session (
                id INTEGER PRIMARY KEY,
                concept TEXT,
                quiz_score REAL,
                engagement_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def track_performance(
        self,
        concept: str,
        quiz_score: float,
        engagement_time: float,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_session (concept, quiz_score, engagement_time)
            VALUES (?, ?, ?)
            """,
            (concept, quiz_score, engagement_time),
        )
        conn.commit()
        conn.close()

    def get_difficulty(self, concept: str) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT quiz_score
            FROM user_session
            WHERE concept = ?
            ORDER BY timestamp DESC, id DESC
            """,
            (concept,),
        )
        scores = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not scores:
            return "moderate"

        latest_score = scores[0]
        if latest_score < 0.6:
            return "basic"
        if latest_score < 0.8:
            return "moderate"
        return "advanced"
