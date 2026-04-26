"""
TEST 1: RAG Setup Module
Location: manual_tests/01_test_rag.py
Run: python manual_tests/01_test_rag.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.rag.rag_setup import (
    load_text,
    split_text,
    load_embedding_model,
    create_embeddings,
    create_faiss_index,
    RAGPipeline
)
import traceback

class TestRAG:
    """Test suite for RAG pipeline"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
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
            self.errors.append((name, str(e), traceback.format_exc()))
            return None
    
    def run_all(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("TEST 1: RAG SETUP MODULE")
        print("="*70)
        
        # Test 1: Load Text
        text = self.test(
            "Load text file",
            lambda: load_text("data/sample.txt")
        )
        
        if text:
            print(f"     Loaded {len(text)} characters")
        
        # Test 2: Split Text
        chunks = self.test(
            "Split text into chunks",
            lambda: split_text(text, chunk_size=150, overlap=10)
        )
        
        if chunks:
            print(f"     Created {len(chunks)} chunks")
        
        # Test 3: Load Model
        model = self.test(
            "Load embedding model",
            lambda: load_embedding_model("all-MiniLM-L6-v2")
        )
        
        if model:
            print(f"     Model loaded successfully")
        
        # Test 4: Create Embeddings
        if chunks and model:
            embeddings = self.test(
                "Create embeddings",
                lambda: create_embeddings(chunks[:10], model, batch_size=4)
            )
            
            if embeddings is not None:
                print(f"     Shape: {embeddings.shape}")
        
        # Test 5: Create FAISS Index
        if embeddings is not None:
            index = self.test(
                "Create FAISS index",
                lambda: create_faiss_index(embeddings, index_type="flat")
            )
            
            if index:
                print(f"     Index with {index.ntotal} vectors")
        
        # Test 6: Full Pipeline
        rag = self.test(
            "Complete RAG pipeline",
            lambda: self._setup_rag()
        )
        
        if rag:
            print(f"     {len(rag.documents)} docs, index ready")
        
        # Test 7: Retrieval
        if rag:
            results = self.test(
                "Retrieve documents",
                lambda: rag.retrieve("What is backpropagation?", k=3)
            )
            
            if results:
                print(f"     Retrieved {len(results)} results")
        
        # Print Summary
        self._print_summary()
        return self.failed == 0
    
    def _setup_rag(self):
        """Helper: Setup RAG pipeline"""
        rag = RAGPipeline()
        rag.setup("data/sample.txt")
        return rag
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {self.passed} ")
        print(f"Failed: {self.failed} ")
        
        if self.errors:
            print(f"\nErrors:")
            for name, error, tb in self.errors:
                print(f"\n  {name}:")
                print(f"    {error}")
                # Uncomment to see full traceback:
                # print(f"    {tb}")
        
        total = self.passed + self.failed
        if total > 0:
            percentage = (self.passed / total) * 100
            print(f"\nSuccess Rate: {percentage:.1f}%")
        
        print("="*70 + "\n")
        
        return self.failed == 0

if __name__ == "__main__":
    tester = TestRAG()
    success = tester.run_all()
    sys.exit(0 if success else 1)



