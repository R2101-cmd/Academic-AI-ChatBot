"""
TEST 3: Graph Builder & Graph Engine
Location: manual_tests/03_test_graph.py
Run: python manual_tests/03_test_graph.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.graph.cognitive_graph import CognitiveGraphEngine
from backend.graph.graph_builder import (
    ConceptExtractor,
    RelationExtractor,
    GraphBuilder
)

class TestGraph:
    """Test suite for Graph operations"""
    
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
            self.errors.append((name, str(e)))
            return None
    
    def run_all(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("TEST 3: GRAPH OPERATIONS")
        print("="*70)
        
        print("\n Testing Cognitive Graph Engine...")
        
        self.test(
            "Create empty graph",
            lambda: self._test_empty_graph()
        )
        
        self.test(
            "Add concepts",
            lambda: self._test_add_concepts()
        )
        
        self.test(
            "Add relations",
            lambda: self._test_add_relations()
        )
        
        self.test(
            "Pathfinding",
            lambda: self._test_pathfinding()
        )
        
        self.test(
            "Build sample graph",
            lambda: self._test_sample_graph()
        )
        
        print("\n Testing Graph Builder...")
        
        self.test(
            "Concept extraction",
            lambda: self._test_concept_extraction()
        )
        
        self.test(
            "Relation extraction",
            lambda: self._test_relation_extraction()
        )
        
        self.test(
            "Build graph from text",
            lambda: self._test_build_from_text()
        )
        
        self._print_summary()
        return self.failed == 0
    
    def _test_empty_graph(self):
        """Test empty graph creation"""
        graph = CognitiveGraphEngine()
        assert graph.graph is not None
        assert graph.graph.number_of_nodes() == 0
    
    def _test_add_concepts(self):
        """Test adding concepts"""
        graph = CognitiveGraphEngine()
        graph.add_concept("Algebra")
        graph.add_concept("Calculus")
        assert graph.graph.number_of_nodes() == 2
        print(f"\n     Added 2 concepts")
    
    def _test_add_relations(self):
        """Test adding relations"""
        graph = CognitiveGraphEngine()
        graph.add_concept("A")
        graph.add_concept("B")
        graph.add_relation("A", "B", "prereq")
        assert graph.graph.has_edge("A", "B")
        print(f"\n     Added 1 relation")
    
    def _test_pathfinding(self):
        """Test pathfinding"""
        graph = CognitiveGraphEngine()
        concepts = ["A", "B", "C", "D"]
        for c in concepts:
            graph.add_concept(c)
        
        graph.add_relation("A", "B", "prereq")
        graph.add_relation("B", "C", "prereq")
        graph.add_relation("C", "D", "prereq")
        
        path = graph.get_path("A", "D")
        assert path == ["A", "B", "C", "D"]
        print(f"\n     Found path: {'  '.join(path)}")
    
    def _test_sample_graph(self):
        """Test sample graph"""
        graph = CognitiveGraphEngine()
        graph.build_sample_graph()
        assert graph.graph.number_of_nodes() > 5
        print(f"\n     Built graph with {graph.graph.number_of_nodes()} nodes, "
              f"{graph.graph.number_of_edges()} edges")
    
    def _test_concept_extraction(self):
        """Test concept extraction"""
        extractor = ConceptExtractor()
        text = "Calculus is the study of change. The Chain Rule is fundamental."
        concepts = extractor.extract_from_text(text)
        assert len(concepts) > 0
        print(f"\n     Extracted {len(concepts)} concepts")
    
    def _test_relation_extraction(self):
        """Test relation extraction"""
        extractor = RelationExtractor()
        text = "Algebra is a prerequisite for Calculus. Calculus is used in Physics."
        relations = extractor.extract_all_relations(text)
        assert len(relations) > 0
        print(f"\n     Found {len(relations)} relations")
    
    def _test_build_from_text(self):
        """Test building graph from text"""
        builder = GraphBuilder()
        texts = [
            "Calculus requires Algebra. Chain Rule is part of Calculus.",
            "Backpropagation uses the Chain Rule. Neural Networks use Backpropagation."
        ]
        graph = builder.build_from_documents(texts)
        assert graph.number_of_nodes() > 0
        print(f"\n     Built graph with {graph.number_of_nodes()} nodes")
    
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
    tester = TestGraph()
    success = tester.run_all()
    sys.exit(0 if success else 1)



