"""
TEST 2: Retriever Agent
Location: manual_tests/02_test_retriever.py
Run: python manual_tests/02_test_retriever.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.rag.rag_setup import RAGPipeline
from backend.agents.retriever_agent import RetrieverAgent
from backend.graph.cognitive_graph import CognitiveGraphEngine
import traceback

class TestRetriever:
    """Test suite for Retriever Agent"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.rag = None
        self.graph = None
        self.retriever = None
    
    def test(self, name, func):
        """Run a test"""
        try:
            print(f"\n   {name}...", end=" ")
            result = func()
            print("")
            self.passed += 1
            return result
        except Exception as e:
            print("")
            self.failed += 1
            self.errors.append((name, str(e)))
            return None
    
    def run_all(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("TEST 2: RETRIEVER AGENT")
        print("="*70)
        
        # Setup
        print("\n Setting up...")
        self.test(
            "Setup RAG pipeline",
            self._setup_rag
        )
        self.test(
            "Build cognitive graph",
            self._setup_graph
        )
        self.test(
            "Initialize retriever",
            self._init_retriever
        )
        
        if not self.retriever:
            print("\n Setup failed. Cannot continue tests.")
            self._print_summary()
            return False
        
        # Tests
        print("\n Testing retriever...")
        
        self.test(
            "Basic retrieval",
            lambda: self._test_basic_retrieval()
        )
        
        self.test(
            "Retrieval returns dict",
            lambda: self._test_returns_dict()
        )
        
        self.test(
            "Graph path extraction",
            lambda: self._test_graph_path()
        )
        
        self.test(
            "Semantic documents",
            lambda: self._test_semantic_docs()
        )
        
        self.test(
            "Combined context",
            lambda: self._test_combined_context()
        )
        
        self.test(
            "Different queries",
            lambda: self._test_different_queries()
        )
        
        self.test(
            "Batch retrieval",
            lambda: self._test_batch_retrieval()
        )
        
        self.test(
            "Index statistics",
            lambda: self._test_index_stats()
        )
        
        self._print_summary()
        return self.failed == 0
    
    def _setup_rag(self):
        """Setup RAG pipeline"""
        self.rag = RAGPipeline()
        self.rag.setup("data/sample.txt")
    
    def _setup_graph(self):
        """Setup graph"""
        self.graph = CognitiveGraphEngine()
        self.graph.build_sample_graph()
    
    def _init_retriever(self):
        """Initialize retriever"""
        self.retriever = RetrieverAgent(
            self.rag.index,
            self.rag.documents,
            self.rag.model,
            self.graph
        )
    
    def _test_basic_retrieval(self):
        """Test basic retrieval"""
        result = self.retriever.retrieve("What is backpropagation?", k=3)
        assert result is not None
        print(f"\n     Retrieved {result['total_unique']} unique docs")
    
    def _test_returns_dict(self):
        """Test returns correct dict"""
        result = self.retriever.retrieve("test query", k=2)
        required_keys = ["query", "semantic_docs", "graph_path", "combined_context"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        print(f"\n     All required keys present")
    
    def _test_graph_path(self):
        """Test graph path extraction"""
        result = self.retriever.retrieve("backpropagation", k=2)
        assert len(result['graph_path']) > 0
        assert "Backpropagation" in result['graph_path']
        print(f"\n     Path: {'  '.join(result['graph_path'])}")
    
    def _test_semantic_docs(self):
        """Test semantic docs"""
        result = self.retriever.retrieve("neural networks", k=3)
        assert len(result['semantic_docs']) > 0
        print(f"\n     Found {len(result['semantic_docs'])} semantic docs")
    
    def _test_combined_context(self):
        """Test combined context"""
        result = self.retriever.retrieve("gradient descent", k=2)
        assert len(result['combined_context']) > 0
        print(f"\n     Context length: {len(result['combined_context'])} chars")
    
    def _test_different_queries(self):
        """Test different queries"""
        queries = ["calculus", "chain rule", "optimization"]
        for query in queries:
            result = self.retriever.retrieve(query, k=1)
            assert result['total_unique'] > 0
        print(f"\n     All {len(queries)} queries succeeded")
    
    def _test_batch_retrieval(self):
        """Test batch retrieval"""
        queries = ["test1", "test2"]
        results = self.retriever.retrieve_batch(queries, k=2)
        assert len(results) == len(queries)
        print(f"\n     Batch processed {len(results)} queries")
    
    def _test_index_stats(self):
        """Test index stats"""
        stats = self.retriever.get_index_stats()
        assert stats['total_documents'] > 0
        print(f"\n     Index: {stats['total_documents']} docs, "
              f"dim: {stats['embedding_dimension']}")
    
    def _print_summary(self):
        """Print summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {self.passed} ")
        print(f"Failed: {self.failed} ")
        
        if self.errors:
            print(f"\nErrors:")
            for name, error in self.errors:
                print(f"   {name}: {error}")
        
        total = self.passed + self.failed
        if total > 0:
            percentage = (self.passed / total) * 100
            print(f"\nSuccess Rate: {percentage:.1f}%")
        
        print("="*70 + "\n")

if __name__ == "__main__":
    tester = TestRetriever()
    success = tester.run_all()
    sys.exit(0 if success else 1)



