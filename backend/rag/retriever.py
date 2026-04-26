"""
Retriever Agent - Hybrid FAISS + Graph-based Retrieval
Combines semantic search with concept graph traversal
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Hybrid retrieval system combining:
    1. FAISS semantic search (dense vector similarity)
    2. Graph-based concept path retrieval (relational structure)
    
    Returns relevant documents + optimal learning path for the query.
    """

    def __init__(
        self,
        faiss_index: faiss.Index,
        documents: List[str],
        model: SentenceTransformer,
        graph_engine=None,
        weights: Dict[str, float] = None
    ):
        """
        Initialize Retriever Agent.
        
        Args:
            faiss_index: FAISS index for semantic search
            documents: List of document chunks
            model: SentenceTransformer for embeddings
            graph_engine: CognitiveGraphEngine for concept paths
            weights: Weighting for hybrid retrieval (default: equal)
        """
        self.index = faiss_index
        self.documents = documents
        self.model = model
        self.graph = graph_engine
        
        # Default weights for hybrid retrieval
        self.weights = weights or {
            "semantic": 0.6,
            "graph": 0.4
        }
        
        logger.info("RetrieverAgent initialized")

    # ============================================================
    # SEMANTIC RETRIEVAL (FAISS)
    # ============================================================

    def _semantic_search(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Retrieve documents using FAISS semantic search.
        
        Args:
            query: Query string
            k: Number of results
        
        Returns:
            (documents, distances)
        """
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            # Retrieve documents
            semantic_docs = []
            semantic_distances = []
            
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    semantic_docs.append(self.documents[idx])
                    semantic_distances.append(float(distance))
            
            logger.debug(f"Semantic search retrieved {len(semantic_docs)} documents")
            return semantic_docs, semantic_distances
        
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return [], []

    # ============================================================
    # GRAPH-BASED RETRIEVAL
    # ============================================================

    def _extract_graph_path(self, query: str) -> Optional[List[str]]:
        """
        Extract learning path from query using concept graph.
        
        Args:
            query: User query
        
        Returns:
            Concept path (e.g., ["Calculus", "Chain Rule", "Backprop"]) or None
        """
        if not self.graph:
            return None
        
        try:
            query_lower = query.lower()
            
            # Concept-to-path mapping (expandable)
            path_map = {
                "backprop": ["Calculus", "Chain Rule", "Backpropagation"],
                "gradient": ["Calculus", "Derivatives", "Gradient"],
                "neural": ["Linear Algebra", "Neural Networks", "Backpropagation"],
                "chain rule": ["Calculus", "Chain Rule", "Derivatives"],
                "derivative": ["Algebra", "Calculus", "Derivatives"],
                "optimization": ["Calculus", "Gradient", "Optimization"],
            }
            
            # Find matching keyword and return path
            for keyword, path in path_map.items():
                if keyword in query_lower:
                    logger.debug(f"Extracted path for '{keyword}': {' → '.join(path)}")
                    return path
            
            # Try to find path between concepts mentioned in query
            # (implement more sophisticated NER here)
            return None
        
        except Exception as e:
            logger.error(f"Graph path extraction failed: {e}")
            return None

    def _graph_based_retrieval(self, graph_path: List[str]) -> List[str]:
        """
        Retrieve documents related to concepts in the graph path.
        
        Args:
            graph_path: List of concepts in learning path
        
        Returns:
            Documents related to the path concepts
        """
        if not graph_path:
            return []
        
        try:
            path_terms = set()
            for concept in graph_path:
                path_terms.add(concept.lower())
            
            # Find documents mentioning path concepts
            relevant_docs = []
            for doc in self.documents:
                doc_lower = doc.lower()
                # Score based on concept mentions
                score = sum(1 for term in path_terms if term in doc_lower)
                if score > 0:
                    relevant_docs.append((doc, score))
            
            # Sort by score and return
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            result = [doc for doc, _ in relevant_docs[:5]]
            
            logger.debug(f"Graph-based retrieval found {len(result)} related documents")
            return result
        
        except Exception as e:
            logger.error(f"Graph-based retrieval failed: {e}")
            return []

    # ============================================================
    # HYBRID RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_graph: bool = True
    ) -> Dict:
        """
        Hybrid retrieval combining semantic search and graph paths.
        
        Args:
            query: User query
            k: Number of semantic results
            use_graph: Whether to use graph-based retrieval
        
        Returns:
            {
                "query": original query,
                "semantic_docs": [list of documents from FAISS],
                "semantic_distances": [similarity distances],
                "graph_path": [concept chain],
                "graph_docs": [documents related to path],
                "combined_docs": [merged results],
                "combined_context": string representation
            }
        """
        logger.info(f"Retrieving for query: '{query}'")
        
        # Semantic retrieval
        semantic_docs, semantic_distances = self._semantic_search(query, k)
        
        # Graph retrieval
        graph_path = None
        graph_docs = []
        
        if use_graph:
            graph_path = self._extract_graph_path(query)
            if graph_path:
                graph_docs = self._graph_based_retrieval(graph_path)
        
        # Default path if none extracted
        if not graph_path:
            graph_path = ["Calculus", "Chain Rule", "Backpropagation"]
        
        # Merge results (semantic + graph)
        combined_docs = list(dict.fromkeys(semantic_docs + graph_docs))  # Deduplicate
        
        # Create combined context
        combined_context = "\n".join(combined_docs)
        
        result = {
            "query": query,
            "semantic_docs": semantic_docs,
            "semantic_distances": semantic_distances,
            "graph_path": graph_path,
            "graph_docs": graph_docs,
            "combined_docs": combined_docs,
            "combined_context": combined_context,
            "num_semantic": len(semantic_docs),
            "num_graph": len(graph_docs),
            "total_unique": len(combined_docs)
        }
        
        logger.info(
            f"Retrieval complete: {len(semantic_docs)} semantic + "
            f"{len(graph_docs)} graph docs = {len(combined_docs)} unique"
        )
        
        return result

    # ============================================================
    # ADVANCED RETRIEVAL OPTIONS
    # ============================================================

    def retrieve_with_reranking(
        self,
        query: str,
        k: int = 10,
        rerank_k: int = 5
    ) -> Dict:
        """
        Retrieve with two-stage reranking.
        
        Args:
            query: Query string
            k: Initial number of candidates
            rerank_k: Final number of results after reranking
        
        Returns:
            Reranked retrieval results
        """
        # First stage: semantic retrieval
        initial_results = self.retrieve(query, k=k)
        
        # Second stage: rerank by cross-encoder (simple: use cosine similarity)
        query_emb = self.model.encode([query])
        
        reranked = []
        for doc in initial_results["combined_docs"][:k]:
            doc_emb = self.model.encode([doc])
            similarity = (query_emb @ doc_emb.T)[0][0]
            reranked.append((doc, similarity))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        final_docs = [doc for doc, _ in reranked[:rerank_k]]
        
        return {
            **initial_results,
            "combined_docs": final_docs,
            "combined_context": "\n".join(final_docs),
            "reranking_used": True
        }

    def retrieve_by_concept(self, concept: str, k: int = 5) -> Dict:
        """
        Retrieve documents related to a specific concept.
        
        Args:
            concept: Concept name
            k: Number of results
        
        Returns:
            Concept-specific retrieval results
        """
        # Find all documents mentioning the concept
        relevant = []
        concept_lower = concept.lower()
        
        for doc in self.documents:
            if concept_lower in doc.lower():
                relevant.append(doc)
        
        # Limit to k results
        relevant = relevant[:k]
        
        return {
            "query": concept,
            "concept": concept,
            "semantic_docs": relevant,
            "combined_docs": relevant,
            "combined_context": "\n".join(relevant),
            "num_results": len(relevant)
        }

    # ============================================================
    # BATCH RETRIEVAL
    # ============================================================

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[Dict]:
        """
        Batch retrieve for multiple queries.
        
        Args:
            queries: List of queries
            k: Results per query
        
        Returns:
            List of retrieval results
        """
        results = []
        for query in queries:
            result = self.retrieve(query, k=k)
            results.append(result)
        
        logger.info(f"Batch retrieval completed for {len(queries)} queries")
        return results

    # ============================================================
    # STATISTICS & DEBUGGING
    # ============================================================

    def get_index_stats(self) -> Dict:
        """Get FAISS index statistics."""
        return {
            "total_documents": len(self.documents),
            "index_type": type(self.index).__name__,
            "index_size": self.index.ntotal,
            "embedding_dimension": self.index.d if hasattr(self.index, 'd') else None,
            "model_name": self.model.get_sentence_embedding_dimension()
        }

    def print_retrieval_analysis(self, result: Dict) -> None:
        """Print detailed analysis of retrieval result."""
        print("\n" + "="*70)
        print(f"Query: {result['query']}")
        print("="*70)
        print(f"Learning Path: {' → '.join(result['graph_path'])}")
        print(f"\nSemantic Results: {result['num_semantic']} documents")
        print(f"Graph-based Results: {result['num_graph']} documents")
        print(f"Total Unique: {result['total_unique']} documents")
        print(f"\nTop Semantic Document:")
        if result['semantic_docs']:
            print(f"  {result['semantic_docs'][0][:100]}...")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage (requires setup)
    from backend.rag.rag_setup import RAGPipeline
    
    # Setup RAG
    rag = RAGPipeline()
    rag.setup("../../data/sample.txt")
    
    # Create retriever
    retriever = RetrieverAgent(
        rag.index,
        rag.documents,
        rag.model
    )
    
    # Test retrieval
    result = retriever.retrieve("Explain backpropagation", k=3)
    retriever.print_retrieval_analysis(result)