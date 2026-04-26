"""
Graph Builder - Automatic construction of cognitive curriculum graphs
Extracts concepts and relations from documents using pattern matching and LLM
"""

import re
from typing import List, Set, Tuple, Dict, Optional
import networkx as nx
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# PATTERN-BASED CONCEPT EXTRACTION
# ============================================================

class ConceptExtractor:
    """
    Extract academic concepts from documents using patterns and heuristics.
    """

    def __init__(self):
        """Initialize concept extractor."""
        self.concepts = set()
        self.concept_mentions = {}

    def extract_from_text(self, text: str) -> Set[str]:
        """
        Extract concepts using multiple heuristics.
        
        Args:
            text: Document text
        
        Returns:
            Set of extracted concepts
        """
        concepts = set()

        # Method 1: Title Case words (2+ words)
        title_case = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
            text
        )
        concepts.update(title_case)

        # Method 2: Words in bold/emphasis (**word**)
        bold_words = re.findall(
            r'\*\*([A-Za-z\s]+)\*\*',
            text
        )
        concepts.update(bold_words)

        # Method 3: Words in code blocks (`word`)
        code_words = re.findall(
            r'`([A-Za-z]+)`',
            text
        )
        concepts.update(code_words)

        # Method 4: Section headers (# Header)
        headers = re.findall(
            r'^#+\s+([A-Za-z\s]+)',
            text,
            re.MULTILINE
        )
        concepts.update(headers)

        # Method 5: Technical terms (common academic keywords)
        technical_terms = [
            'Algorithm', 'Matrix', 'Vector', 'Function', 'Theorem',
            'Model', 'Network', 'Layer', 'Activation', 'Loss',
            'Optimization', 'Regularization', 'Convergence', 'Gradient',
            'Backpropagation', 'Derivative', 'Integral', 'Probability',
            'Distribution', 'Parameter', 'Hyperparameter', 'Validation'
        ]
        
        for term in technical_terms:
            if re.search(r'\b' + term + r'\b', text, re.IGNORECASE):
                concepts.add(term)

        # Filter: Remove very short concepts and clean
        concepts = {
            c.strip() for c in concepts
            if len(c.strip()) > 2 and c.strip()
        }

        logger.debug(f"Extracted {len(concepts)} concepts from text")
        return concepts

    def extract_from_documents(self, documents: List[str]) -> Set[str]:
        """
        Extract concepts from multiple documents.
        
        Args:
            documents: List of document strings
        
        Returns:
            Set of all extracted concepts
        """
        all_concepts = set()

        for doc in documents:
            concepts = self.extract_from_text(doc)
            all_concepts.update(concepts)
            
            # Track mentions
            for concept in concepts:
                if concept not in self.concept_mentions:
                    self.concept_mentions[concept] = 0
                self.concept_mentions[concept] += 1

        logger.info(f"Extracted {len(all_concepts)} unique concepts")
        return all_concepts


# ============================================================
# RELATION EXTRACTION
# ============================================================

class RelationExtractor:
    """
    Extract relationships between concepts from documents.
    """

    def __init__(self):
        """Initialize relation extractor."""
        self.relations = []

    def extract_prerequisites(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract prerequisite relations using patterns.
        
        Examples:
            "X is a prerequisite for Y"
            "X is required for Y"
            "To understand Y, first learn X"
        
        Args:
            text: Document text
        
        Returns:
            List of (source, target) tuples
        """
        relations = []

        patterns = [
            # "X is a prerequisite for Y"
            (r'(\w+)\s+is\s+a\s+prerequisite\s+(?:for|of)\s+(\w+)', 'prereq'),
            # "X is required for Y"
            (r'(\w+)\s+is\s+required\s+(?:for|to)\s+(\w+)', 'prereq'),
            # "X is needed for Y"
            (r'(\w+)\s+is\s+needed\s+(?:for|to)\s+(\w+)', 'prereq'),
            # "Before learning Y, learn X"
            (r'(?:Before|To)\s+(?:learning|understanding)\s+(\w+),\s+(?:learn|understand)\s+(\w+)', 'prereq'),
        ]

        for pattern, rel_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                relations.append((match[0], match[1], rel_type))

        return relations

    def extract_applications(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract application relations.
        
        Examples:
            "X is used in Y"
            "Y applies X"
        """
        relations = []

        patterns = [
            (r'(\w+)\s+is\s+used\s+in\s+(\w+)', 'applies_to'),
            (r'(\w+)\s+applies\s+(\w+)', 'applies_to'),
            (r'(\w+)\s+enables\s+(\w+)', 'enables'),
        ]

        for pattern, rel_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                relations.append((match[0], match[1], rel_type))

        return relations

    def extract_similarity(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract similarity relations.
        
        Examples:
            "X is similar to Y"
            "X and Y are related"
        """
        relations = []

        patterns = [
            (r'(\w+)\s+is\s+similar\s+to\s+(\w+)', 'similar'),
            (r'(\w+)\s+and\s+(\w+)\s+are\s+related', 'related'),
        ]

        for pattern, rel_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                relations.append((match[0], match[1], rel_type))

        return relations

    def extract_all_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract all types of relations from text.
        
        Args:
            text: Document text
        
        Returns:
            List of (source, target, relation_type) tuples
        """
        relations = []
        relations.extend(self.extract_prerequisites(text))
        relations.extend(self.extract_applications(text))
        relations.extend(self.extract_similarity(text))

        logger.debug(f"Extracted {len(relations)} relations")
        return relations


# ============================================================
# SEMANTIC RELATION EXTRACTION (using embeddings)
# ============================================================

class SemanticRelationExtractor:
    """
    Extract relations using semantic similarity (embeddings).
    """

    def __init__(self, model: SentenceTransformer, threshold: float = 0.7):
        """
        Initialize semantic extractor.
        
        Args:
            model: SentenceTransformer for embeddings
            threshold: Similarity threshold for relations
        """
        self.model = model
        self.threshold = threshold

    def find_semantic_relations(
        self,
        concepts: List[str],
        documents: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Find semantic relations between concepts.
        
        Args:
            concepts: List of concepts
            documents: List of documents (context)
        
        Returns:
            List of (source, target, similarity) tuples
        """
        relations = []

        try:
            # Create concept embeddings
            concept_embeddings = self.model.encode(concepts)

            # Compute pairwise similarities
            for i, concept1 in enumerate(concepts):
                for j, concept2 in enumerate(concepts):
                    if i >= j:  # Skip self and duplicates
                        continue

                    # Cosine similarity
                    similarity = (
                        concept_embeddings[i] @ concept_embeddings[j] /
                        (np.linalg.norm(concept_embeddings[i]) *
                         np.linalg.norm(concept_embeddings[j]) + 1e-8)
                    )

                    # Only include above threshold
                    if similarity > self.threshold:
                        relations.append((concept1, concept2, float(similarity)))

            logger.info(f"Found {len(relations)} semantic relations")
            return relations

        except Exception as e:
            logger.error(f"Semantic relation extraction failed: {e}")
            return []


# ============================================================
# GRAPH BUILDER
# ============================================================

class GraphBuilder:
    """
    Build academic curriculum graphs from documents.
    Combines pattern matching and semantic similarity.
    """

    def __init__(self):
        """Initialize graph builder."""
        self.concept_extractor = ConceptExtractor()
        self.relation_extractor = RelationExtractor()
        self.semantic_extractor = None
        self.graph = nx.DiGraph()

    def build_from_documents(
        self,
        documents: List[str],
        use_semantic: bool = False,
        embedding_model: Optional[SentenceTransformer] = None
    ) -> nx.DiGraph:
        """
        Build curriculum graph from documents.
        
        Args:
            documents: List of document strings
            use_semantic: Whether to use semantic similarity
            embedding_model: Model for semantic extraction
        
        Returns:
            NetworkX directed graph
        """
        logger.info("Building curriculum graph from documents...")

        # Step 1: Extract concepts
        concepts = self.concept_extractor.extract_from_documents(documents)
        logger.info(f"Extracted {len(concepts)} concepts")

        # Step 2: Add concept nodes
        for concept in concepts:
            self.graph.add_node(concept)

        # Step 3: Extract pattern-based relations
        all_relations = []
        combined_text = "\n".join(documents)

        pattern_relations = self.relation_extractor.extract_all_relations(combined_text)
        all_relations.extend(pattern_relations)
        logger.info(f"Extracted {len(pattern_relations)} pattern-based relations")

        # Step 4: Extract semantic relations (optional)
        if use_semantic and embedding_model:
            self.semantic_extractor = SemanticRelationExtractor(embedding_model)
            semantic_relations = self.semantic_extractor.find_semantic_relations(
                list(concepts),
                documents
            )
            
            # Add only high-confidence semantic relations
            semantic_relations = [
                (src, tgt, "semantic") for src, tgt, sim in semantic_relations
                if sim > 0.8
            ]
            all_relations.extend(semantic_relations)
            logger.info(f"Extracted {len(semantic_relations)} semantic relations")

        # Step 5: Add relations to graph
        valid_relations = 0
        for source, target, rel_type in all_relations:
            if source in self.graph and target in self.graph:
                self.graph.add_edge(source, target, relation=rel_type)
                valid_relations += 1

        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and "
                   f"{self.graph.number_of_edges()} edges ({valid_relations} valid relations)")

        return self.graph

    def build_sample_graph(self) -> nx.DiGraph:
        """
        Build a hand-crafted sample graph for testing.
        
        Returns:
            Sample curriculum graph
        """
        # Define concepts
        concepts = [
            "Algebra", "Calculus", "Chain Rule", "Linear Algebra",
            "Derivatives", "Gradient", "Neural Networks",
            "Backpropagation", "Optimization", "Activation Functions"
        ]

        for concept in concepts:
            self.graph.add_node(concept)

        # Define relations
        relations = [
            ("Algebra", "Calculus", "prereq"),
            ("Calculus", "Derivatives", "prereq"),
            ("Derivatives", "Chain Rule", "prereq"),
            ("Calculus", "Chain Rule", "applies_to"),
            ("Chain Rule", "Gradient", "applies_to"),
            ("Gradient", "Backpropagation", "prereq"),
            ("Linear Algebra", "Neural Networks", "prereq"),
            ("Neural Networks", "Backpropagation", "applies_to"),
            ("Gradient", "Optimization", "applies_to"),
            ("Backpropagation", "Neural Networks", "applies_to"),
            ("Activation Functions", "Neural Networks", "applies_to"),
        ]

        for source, target, rel_type in relations:
            self.graph.add_edge(source, target, relation=rel_type)

        logger.info("Built sample curriculum graph")
        return self.graph

    # ============================================================
    # GRAPH ANALYSIS & VISUALIZATION
    # ============================================================

    def get_graph_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "density": nx.density(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
        }

    def export_to_json(self, filepath: str) -> None:
        """Export graph to JSON format."""
        import json

        data = {
            "nodes": list(self.graph.nodes()),
            "edges": [
                {
                    "source": src,
                    "target": tgt,
                    "relation": self.graph.edges[src, tgt].get("relation", "unknown")
                }
                for src, tgt in self.graph.edges()
            ],
            "stats": self.get_graph_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Graph exported to {filepath}")

    def print_graph_info(self) -> None:
        """Print graph information."""
        stats = self.get_graph_stats()
        
        print("\n" + "="*70)
        print("CURRICULUM GRAPH STATISTICS")
        print("="*70)
        print(f"Concepts (Nodes): {stats['num_nodes']}")
        print(f"Relations (Edges): {stats['num_edges']}")
        print(f"Is DAG: {stats['is_dag']}")
        print(f"Graph Density: {stats['density']:.3f}")
        print(f"Average Degree: {stats['avg_degree']:.2f}")
        print("\nTop Concepts (by degree):")
        
        degrees = dict(self.graph.degree())
        sorted_concepts = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        for concept, degree in sorted_concepts[:10]:
            print(f"  {concept}: {degree}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    builder = GraphBuilder()
    
    # Build sample graph
    graph = builder.build_sample_graph()
    builder.print_graph_info()
    
    # Export graph
    builder.export_to_json("curriculum_graph.json")