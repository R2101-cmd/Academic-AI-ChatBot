"""Hybrid retrieval agent for semantic search plus Graph-CoT guidance."""

from __future__ import annotations

from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RetrieverAgent:
    """Combine FAISS retrieval with graph-aware concept path selection."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        documents: List[str],
        model: SentenceTransformer,
        graph_engine=None,
    ) -> None:
        self.index = faiss_index
        self.documents = documents
        self.model = model
        self.graph = graph_engine

    def _semantic_search(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
        if self.index.ntotal == 0 or not self.documents:
            return [], []

        normalized_query = self._normalize_query(query)
        query_embedding = self.model.encode([normalized_query], convert_to_numpy=True)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        top_k = max(1, min(k, self.index.ntotal))
        distances, indices = self.index.search(query_embedding, top_k)

        docs: List[str] = []
        scores: List[float] = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.documents):
                docs.append(self.documents[idx])
                scores.append(float(distance))
        return docs, scores

    def _normalize_query(self, query: str) -> str:
        """Normalize common phrasing and spelling variants before retrieval."""
        normalized = query.lower().strip()
        replacements = {
            "artificial intelligent": "artificial intelligence",
            "ai technology": "artificial intelligence",
            "ml": "machine learning",
            "nlp": "natural language processing",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _extract_graph_path(self, query: str) -> List[str]:
        if not self.graph:
            return self._default_graph_path(query)

        query_lower = query.lower()
        known_concepts = [
            concept
            for concept in self.graph.graph.nodes()
            if concept.lower() in query_lower
        ]

        if len(known_concepts) >= 2:
            path = self.graph.get_path(known_concepts[0], known_concepts[-1], max_hops=10)
            if path:
                return path

        if len(known_concepts) == 1:
            concept = known_concepts[0]
            chain = self._build_prerequisite_chain(concept)
            if chain:
                return chain
            return [concept]

        return self._default_graph_path(query)

    def _build_prerequisite_chain(self, concept: str, limit: int = 5) -> List[str]:
        """Walk backwards through prerequisite edges to build a learning path."""
        chain = [concept]
        current = concept

        for _ in range(limit):
            predecessors = list(self.graph.graph.predecessors(current))
            if not predecessors:
                break

            preferred = None
            for predecessor in predecessors:
                relation = self.graph.graph.edges[predecessor, current].get("relation")
                if relation == "prereq":
                    preferred = predecessor
                    break

            next_concept = preferred or predecessors[0]
            chain.insert(0, next_concept)
            current = next_concept

        return chain

    def _default_graph_path(self, query: str) -> List[str]:
        query_lower = self._normalize_query(query)
        fallback_map = {
            "artificial intelligence": ["Computer Science", "Artificial Intelligence", "Applications"],
            "machine learning": ["Statistics", "Machine Learning", "Applications"],
            "backprop": ["Calculus", "Derivatives", "Chain Rule", "Gradient", "Backpropagation"],
            "gradient": ["Calculus", "Derivatives", "Gradient"],
            "neural": ["Linear Algebra", "Neural Networks", "Backpropagation"],
            "chain rule": ["Calculus", "Derivatives", "Chain Rule"],
            "optimization": ["Calculus", "Gradient", "Optimization"],
        }
        for keyword, path in fallback_map.items():
            if keyword in query_lower:
                return path
        return ["Foundations", "Core Concept", "Application"]

    def _graph_based_retrieval(self, graph_path: List[str], limit: int = 5) -> List[str]:
        if not graph_path:
            return []

        concept_terms = {concept.lower() for concept in graph_path}
        ranked_docs: List[Tuple[str, int]] = []
        for doc in self.documents:
            doc_lower = doc.lower()
            score = sum(1 for term in concept_terms if term in doc_lower)
            if score:
                ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:limit]]

    def retrieve(self, query: str, k: int = 5, use_graph: bool = True) -> Dict:
        semantic_docs, semantic_distances = self._semantic_search(query, k=k)
        graph_path = self._extract_graph_path(query) if use_graph else self._default_graph_path(query)
        graph_docs = self._graph_based_retrieval(graph_path, limit=k) if use_graph else []

        combined_docs = list(dict.fromkeys(semantic_docs + graph_docs))
        combined_context = "\n\n".join(combined_docs)

        return {
            "query": query,
            "semantic_docs": semantic_docs,
            "semantic_distances": semantic_distances,
            "graph_path": graph_path,
            "graph_docs": graph_docs,
            "combined_docs": combined_docs,
            "combined_context": combined_context,
            "num_semantic": len(semantic_docs),
            "num_graph": len(graph_docs),
            "total_unique": len(combined_docs),
        }

    def retrieve_batch(self, queries: List[str], k: int = 5) -> List[Dict]:
        return [self.retrieve(query, k=k) for query in queries]

    def retrieve_by_concept(self, concept: str, k: int = 5) -> Dict:
        matches = [doc for doc in self.documents if concept.lower() in doc.lower()][:k]
        return {
            "query": concept,
            "concept": concept,
            "semantic_docs": matches,
            "combined_docs": matches,
            "combined_context": "\n\n".join(matches),
            "num_results": len(matches),
        }

    def get_index_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "index_type": type(self.index).__name__,
            "index_size": self.index.ntotal,
            "embedding_dimension": getattr(self.index, "d", None),
            "model_embedding_dimension": self.model.get_sentence_embedding_dimension(),
        }
