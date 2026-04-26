"""
RAG (Retrieval-Augmented Generation) Module
Handles document loading, splitting, embedding
"""

from backend.rag.rag_setup import load_text, split_text, create_faiss_index

__all__ = ["load_text", "split_text", "create_faiss_index"]