"""Lightweight in-memory session state for conversational tutoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class SessionState:
    user_id: str
    latest_topic: Optional[str] = None
    latest_graph_path: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class SessionManager:
    """Track the current learning topic without external services."""

    def __init__(self, max_history: int = 8) -> None:
        self.max_history = max_history
        self._sessions: Dict[str, SessionState] = {}

    def get(self, user_id: str) -> SessionState:
        key = user_id or "default"
        if key not in self._sessions:
            self._sessions[key] = SessionState(user_id=key)
        return self._sessions[key]

    def update_topic(self, user_id: str, topic: str, graph_path: List[str]) -> None:
        session = self.get(user_id)
        clean_topic = topic.strip()
        if clean_topic:
            session.latest_topic = clean_topic
        session.latest_graph_path = list(dict.fromkeys(graph_path))
        session.updated_at = datetime.utcnow()

    def add_turn(self, user_id: str, role: str, content: str) -> None:
        session = self.get(user_id)
        session.history.append({"role": role, "content": content.strip()})
        session.history = session.history[-self.max_history :]
        session.updated_at = datetime.utcnow()

    def complete_exchange(self, user_id: str, user: str, assistant: str, topic: str) -> None:
        session = self.get(user_id)
        session.chat_history.append(
            {
                "user": user.strip(),
                "assistant": assistant.strip(),
                "topic": topic.strip(),
            }
        )
        session.chat_history = session.chat_history[-self.max_history :]
        session.updated_at = datetime.utcnow()
