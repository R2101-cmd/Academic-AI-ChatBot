"""SentenceTransformer + FAISS retrieval module."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import BACKEND_DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL
from backend.rag.rag_setup import FallbackEmbeddingModel


class RetrievalModule:
    """Load academic notes, embed them, and retrieve relevant chunks."""

    def __init__(self, data_dir: Path = DATA_DIR, model_name: str = EMBEDDING_MODEL) -> None:
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.documents: list[dict[str, Any]] = []
        self.model: Any = None
        self.index: faiss.Index | None = None
        self._build()

    def _build(self) -> None:
        text_blocks = self._load_text_blocks()
        self.documents = self._chunk_blocks(text_blocks)
        self.model = self._load_model()
        embeddings = self.model.encode(
            [doc["text"] for doc in self.documents],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _load_model(self) -> Any:
        try:
            cache_dir = BACKEND_DATA_DIR / "hf_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_dir))
            return SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            return FallbackEmbeddingModel()

    def _load_text_blocks(self) -> list[tuple[str, str]]:
        blocks: list[tuple[str, str]] = []
        for file_path in sorted(self.data_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            blocks.append((file_path.stem.replace("_", " ").title(), text))
        if not blocks:
            blocks.append(("Sample Academic Notes", "Academic learning uses concepts, examples, quizzes, and revision."))
        return blocks

    def _chunk_blocks(self, blocks: list[tuple[str, str]]) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        for title, text in blocks:
            clean_text = re.sub(r"\s+", " ", text).strip()
            for start in range(0, len(clean_text), step):
                chunk = clean_text[start:start + CHUNK_SIZE].strip()
                if len(chunk) >= 40:
                    docs.append({"title": title, "text": chunk})
        return docs

    def retrieve(self, query: str, k: int = 5) -> dict[str, Any]:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for score, index in zip(scores[0], indices[0]):
            if 0 <= index < len(self.documents):
                doc = self.documents[index]
                results.append({
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(max(0.0, score)),
                    "concepts": self._concepts_from_text(doc["text"], query),
                })

        return {
            "query": query,
            "results": results,
            "combined_context": "\n\n".join(item["text"] for item in results),
            "total_unique": len(results),
            "embedding_model": self.model_name,
            "index_type": type(self.index).__name__,
        }

    def _concepts_from_text(self, text: str, query: str) -> list[str]:
        words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[a-zA-Z]{5,}", text)
        query_terms = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
        concepts = []
        for word in words:
            normalized = word.strip().title()
            if normalized.lower() in query_terms or len(concepts) < 3:
                concepts.append(normalized)
        return list(dict.fromkeys(concepts))[:5]

    def stats(self) -> dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.index.d if self.index else None,
            "embedding_model": self.model_name,
        }
