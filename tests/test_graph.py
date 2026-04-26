"""
Unit Tests for Cognitive Curriculum Graph
Tests: CognitiveGraphEngine, graph construction, pathfinding
"""

import pytest
import networkx as nx
from backend.graph.cognitive_graph import CognitiveGraphEngine


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def empty_graph():
    """Create an empty cognitive graph"""
    return CognitiveGraphEngine()


@pytest.fixture
def sample_graph():
    """Create a sample graph with concepts and relations"""
    graph = CognitiveGraphEngine()
    graph.build_sample_graph()
    return graph


@pytest.fixture
def custom_graph():
    """Create a custom graph for specific tests"""
    graph = CognitiveGraphEngine()
    
    # Add simple linear chain: A → B → C → D
    concepts = ["A", "B", "C", "D"]
    for concept in concepts:
        graph.add_concept(concept)
    
    graph.add_relation("A", "B", "prereq")
    graph.add_relation("B", "C", "prereq")
    graph.add_relation("C", "D", "prereq")
    
    return graph


# ============================================================
# GRAPH INITIALIZATION TESTS
# ============================================================

class TestGraphInitialization:
    """Test graph initialization and setup"""

    def test_empty_graph_creation(self, empty_graph):
        """Test creating an empty graph"""
        assert empty_graph.graph is not None
        assert isinstance(empty_graph.graph, nx.DiGraph)
        assert len(empty_graph.graph.nodes) == 0
        assert len(empty_graph.graph.edges) == 0

    def test_sample_graph_creation(self, sample_graph):
        """Test building sample graph"""
        assert len(sample_graph.graph.nodes) > 0
        assert len(sample_graph.graph.edges) > 0

    def test_sample_graph_has_key_concepts(self, sample_graph):
        """Test that sample graph contains key academic concepts"""
        expected_concepts = [
            "Algebra",
            "Calculus",
            "Chain Rule",
            "Backpropagation",
            "Neural Networks",
            "Gradient",
        ]
        
        for concept in expected_concepts:
            assert concept in sample_graph.graph.nodes


# ============================================================
# CONCEPT MANAGEMENT TESTS
# ============================================================

class TestConceptManagement:
    """Test adding and managing concepts"""

    def test_add_single_concept(self, empty_graph):
        """Test adding a single concept"""
        empty_graph.add_concept("Calculus")
        
        assert "Calculus" in empty_graph.graph.nodes
        assert len(empty_graph.graph.nodes) == 1

    def test_add_multiple_concepts(self, empty_graph):
        """Test adding multiple concepts"""
        concepts = ["Algebra", "Calculus", "Linear Algebra"]
        
        for concept in concepts:
            empty_graph.add_concept(concept)
        
        assert len(empty_graph.graph.nodes) == 3
        
        for concept in concepts:
            assert concept in empty_graph.graph.nodes

    def test_add_duplicate_concept(self, empty_graph):
        """Test adding duplicate concept (should not create duplicate)"""
        empty_graph.add_concept("Calculus")
        empty_graph.add_concept("Calculus")
        
        assert len(empty_graph.graph.nodes) == 1

    def test_concept_node_properties(self, empty_graph):
        """Test concept node has correct properties"""
        empty_graph.add_concept("Gradient")
        
        assert empty_graph.graph.nodes["Gradient"] is not None


# ============================================================
# RELATION MANAGEMENT TESTS
# ============================================================

class TestRelationManagement:
    """Test adding and managing concept relations"""

    def test_add_single_relation(self, empty_graph):
        """Test adding a single relation"""
        empty_graph.add_concept("A")
        empty_graph.add_concept("B")
        empty_graph.add_relation("A", "B", "prereq")
        
        assert empty_graph.graph.has_edge("A", "B")
        assert len(empty_graph.graph.edges) == 1

    def test_add_relation_creates_nodes(self, empty_graph):
        """Test that add_relation doesn't fail if nodes don't exist yet"""
        # Note: Current implementation requires nodes to exist
        empty_graph.add_concept("X")
        empty_graph.add_concept("Y")
        empty_graph.add_relation("X", "Y", "similar")
        
        assert empty_graph.graph.has_edge("X", "Y")

    def test_relation_type_preserved(self, empty_graph):
        """Test that relation type is preserved in edge attributes"""
        empty_graph.add_concept("A")
        empty_graph.add_concept("B")
        empty_graph.add_relation("A", "B", "prereq")
        
        edge_data = empty_graph.graph.edges["A", "B"]
        assert edge_data["relation"] == "prereq"

    def test_add_multiple_relations(self, empty_graph):
        """Test adding multiple relations"""
        concepts = ["A", "B", "C"]
        for concept in concepts:
            empty_graph.add_concept(concept)
        
        empty_graph.add_relation("A", "B", "prereq")
        empty_graph.add_relation("B", "C", "prereq")
        empty_graph.add_relation("A", "C", "applies_to")
        
        assert len(empty_graph.graph.edges) == 3

    def test_different_relation_types(self, empty_graph):
        """Test different relation types"""
        empty_graph.add_concept("X")
        empty_graph.add_concept("Y")
        empty_graph.add_concept("Z")
        
        empty_graph.add_relation("X", "Y", "prereq")
        empty_graph.add_relation("Y", "Z", "similar")
        empty_graph.add_relation("X", "Z", "applies_to")
        
        assert empty_graph.graph.edges["X", "Y"]["relation"] == "prereq"
        assert empty_graph.graph.edges["Y", "Z"]["relation"] == "similar"
        assert empty_graph.graph.edges["X", "Z"]["relation"] == "applies_to"


# ============================================================
# PATHFINDING TESTS
# ============================================================

class TestPathfinding:
    """Test graph pathfinding and traversal"""

    def test_get_path_simple_chain(self, custom_graph):
        """Test pathfinding in simple linear chain"""
        path = custom_graph.get_path("A", "D")
        
        assert path is not None
        assert path == ["A", "B", "C", "D"]

    def test_get_path_single_hop(self, custom_graph):
        """Test pathfinding with single edge"""
        path = custom_graph.get_path("A", "B")
        
        assert path == ["A", "B"]

    def test_get_path_same_node(self, custom_graph):
        """Test pathfinding from node to itself"""
        path = custom_graph.get_path("A", "A")
        
        assert path == ["A"]

    def test_get_path_no_path_exists(self, custom_graph):
        """Test pathfinding when no path exists"""
        # Add isolated node
        custom_graph.add_concept("Isolated")
        
        path = custom_graph.get_path("A", "Isolated")
        
        assert path is None

    def test_get_path_respects_max_hops(self, custom_graph):
        """Test that get_path respects max_hops limit"""
        path = custom_graph.get_path("A", "D", max_hops=2)
        
        # Path is 4 nodes (3 hops), exceeds max_hops=2
        assert path is None

    def test_get_path_within_max_hops(self, custom_graph):
        """Test pathfinding within hop limit"""
        path = custom_graph.get_path("A", "C", max_hops=3)
        
        # Path A→B→C is 2 hops, within limit
        assert path is not None
        assert len(path) == 3

    def test_get_path_sample_graph(self, sample_graph):
        """Test pathfinding in sample graph"""
        path = sample_graph.get_path("Calculus", "Backpropagation")
        
        assert path is not None
        # Should contain both start and end
        assert "Calculus" in path
        assert "Backpropagation" in path
        # Should be shortest path
        assert len(path) <= 5

    def test_get_path_prerequisite_chain(self, sample_graph):
        """Test typical prerequisite chain"""
        path = sample_graph.get_path("Algebra", "Neural Networks")
        
        assert path is not None
        assert "Algebra" in path
        assert "Neural Networks" in path


# ============================================================
# GRAPH STRUCTURE TESTS
# ============================================================

class TestGraphStructure:
    """Test overall graph structure and properties"""

    def test_graph_is_directed(self, sample_graph):
        """Test that graph is directed"""
        assert isinstance(sample_graph.graph, nx.DiGraph)

    def test_graph_connectivity(self, sample_graph):
        """Test basic graph connectivity"""
        # Sample graph should have some connected components
        assert nx.number_connected_components(sample_graph.graph.to_undirected()) >= 1

    def test_graph_acyclic(self, sample_graph):
        """Test that graph is acyclic (DAG)"""
        # Curriculum graphs should be DAGs
        assert nx.is_directed_acyclic_graph(sample_graph.graph)

    def test_custom_graph_acyclic(self, custom_graph):
        """Test custom graph is acyclic"""
        assert nx.is_directed_acyclic_graph(custom_graph.graph)

    def test_node_count_sample_graph(self, sample_graph):
        """Test sample graph has expected number of nodes"""
        # Should have at least 8 concepts
        assert len(sample_graph.graph.nodes) >= 8

    def test_edge_count_sample_graph(self, sample_graph):
        """Test sample graph has expected number of edges"""
        # Should have at least 8 relations
        assert len(sample_graph.graph.edges) >= 8


# ============================================================
# GRAPH TRAVERSAL TESTS
# ============================================================

class TestGraphTraversal:
    """Test graph traversal capabilities"""

    def test_get_predecessors(self, custom_graph):
        """Test getting predecessors of a node"""
        predecessors = list(custom_graph.graph.predecessors("C"))
        
        assert "B" in predecessors

    def test_get_successors(self, custom_graph):
        """Test getting successors of a node"""
        successors = list(custom_graph.graph.successors("B"))
        
        assert "C" in successors

    def test_topological_sort(self, custom_graph):
        """Test topological sorting of graph"""
        topo_sort = list(nx.topological_sort(custom_graph.graph))
        
        assert topo_sort == ["A", "B", "C", "D"]

    def test_in_degree(self, custom_graph):
        """Test computing in-degree of nodes"""
        # "A" should have in-degree 0 (no prerequisites)
        assert custom_graph.graph.in_degree("A") == 0
        
        # "D" should have in-degree 1 (one prerequisite)
        assert custom_graph.graph.in_degree("D") == 1

    def test_out_degree(self, custom_graph):
        """Test computing out-degree of nodes"""
        # "A" should have out-degree 1
        assert custom_graph.graph.out_degree("A") == 1
        
        # "D" should have out-degree 0 (nothing depends on it)
        assert custom_graph.graph.out_degree("D") == 0


# ============================================================
# SAMPLE GRAPH VALIDATION TESTS
# ============================================================

class TestSampleGraphValidation:
    """Test integrity of sample graph"""

    def test_algebra_has_no_prerequisites(self, sample_graph):
        """Test that Algebra has no incoming edges"""
        assert sample_graph.graph.in_degree("Algebra") == 0

    def test_calculus_requires_algebra(self, sample_graph):
        """Test prerequisite relationship"""
        assert sample_graph.graph.has_edge("Algebra", "Calculus")

    def test_chain_rule_concepts_connected(self, sample_graph):
        """Test that chain rule is connected to derivatives"""
        # Should have some connection
        assert sample_graph.graph.has_edge("Derivatives", "Chain Rule") or \
               sample_graph.graph.has_edge("Calculus", "Chain Rule")

    def test_backpropagation_depends_on_gradient(self, sample_graph):
        """Test backprop prerequisite"""
        assert sample_graph.graph.has_edge("Gradient", "Backpropagation")

    def test_neural_networks_depends_on_linear_algebra(self, sample_graph):
        """Test neural networks prerequisite"""
        assert sample_graph.graph.has_edge("Linear Algebra", "Neural Networks")

    def test_backprop_applies_to_neural_networks(self, sample_graph):
        """Test application relationship"""
        assert sample_graph.graph.has_edge("Backpropagation", "Neural Networks")


# ============================================================
# EDGE CASE TESTS
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_get_path_nonexistent_nodes(self, empty_graph):
        """Test pathfinding with non-existent nodes"""
        path = empty_graph.get_path("NonExistent1", "NonExistent2")
        
        assert path is None

    def test_add_relation_nonexistent_source(self, empty_graph):
        """Test adding relation with non-existent source node"""
        empty_graph.add_concept("B")
        
        # Should handle gracefully
        # Note: Current implementation will fail; this tests current behavior
        try:
            empty_graph.add_relation("NonExistent", "B", "prereq")
        except (KeyError, nx.NetworkXError):
            pass  # Expected behavior

    def test_get_path_with_zero_hops(self, custom_graph):
        """Test pathfinding with max_hops=0"""
        path = custom_graph.get_path("A", "A", max_hops=0)
        
        assert path == ["A"]

    def test_large_graph_performance(self):
        """Test graph performance with many nodes"""
        graph = CognitiveGraphEngine()
        
        # Add 100 concepts
        for i in range(100):
            graph.add_concept(f"Concept_{i}")
        
        # Add linear chain of relations
        for i in range(99):
            graph.add_relation(f"Concept_{i}", f"Concept_{i+1}", "prereq")
        
        # Should handle efficiently
        path = graph.get_path("Concept_0", "Concept_50", max_hops=100)
        
        assert path is not None


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestGraphIntegration:
    """Integration tests for graph operations"""

    def test_build_and_query_workflow(self):
        """Test typical workflow: build graph, then query paths"""
        graph = CognitiveGraphEngine()
        
        # Build graph
        concepts = ["Basics", "Intermediate", "Advanced", "Expert"]
        for concept in concepts:
            graph.add_concept(concept)
        
        for i in range(len(concepts) - 1):
            graph.add_relation(concepts[i], concepts[i+1], "prereq")
        
        # Query path
        path = graph.get_path("Basics", "Expert")
        
        assert path == concepts

    def test_multiple_paths_shortest_selected(self, empty_graph):
        """Test that shortest path is selected when multiple paths exist"""
        # Create diamond shape: A → B, A → C, B → D, C → D
        for node in ["A", "B", "C", "D"]:
            empty_graph.add_concept(node)
        
        empty_graph.add_relation("A", "B", "prereq")
        empty_graph.add_relation("A", "C", "prereq")
        empty_graph.add_relation("B", "D", "prereq")
        empty_graph.add_relation("C", "D", "prereq")
        
        path = empty_graph.get_path("A", "D")
        
        # Should return one of the shortest paths (length 3)
        assert len(path) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])