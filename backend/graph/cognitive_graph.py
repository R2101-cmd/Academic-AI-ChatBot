"""
Cognitive Curriculum Graph Engine
Manages concept relationships and learning paths
"""

import networkx as nx
from typing import List, Optional


class CognitiveGraphEngine:
    """Build and manage academic concept graph"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_concept(self, concept: str) -> None:
        """Add concept node"""
        clean_concept = concept.strip()
        if clean_concept:
            self.graph.add_node(clean_concept)

    def add_relation(self, source: str, target: str, relation_type: str) -> None:
        """
        Add relation between concepts
        relation_type: 'prereq' | 'similar' | 'applies_to'
        """
        source = source.strip()
        target = target.strip()
        if source and target and source != target:
            self.graph.add_edge(source, target, relation=relation_type)

    def _clean_path(self, path: List[str], max_hops: int) -> Optional[List[str]]:
        clean_path: List[str] = []
        seen = set()
        for concept in path:
            if not concept or concept in seen:
                continue
            clean_path.append(concept)
            seen.add(concept)
        hops = max(len(clean_path) - 1, 0)
        if not clean_path or hops > max_hops:
            return None
        return clean_path

    def get_path(self, start: str, end: str, max_hops: int = 6) -> Optional[List[str]]:
        """Get shortest path between concepts"""
        try:
            if start == end:
                return [start]
            path = nx.shortest_path(self.graph, start, end)
            return self._clean_path(path, max_hops)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def build_sample_graph(self) -> None:
        """Build sample curriculum graph"""
        concepts = [
            "Algebra",
            "Calculus",
            "Chain Rule",
            "Linear Algebra",
            "Derivatives",
            "Gradient",
            "Neural Networks",
            "Backpropagation",
        ]

        for concept in concepts:
            self.add_concept(concept)

        relations = [
            ("Algebra", "Calculus", "prereq"),
            ("Calculus", "Derivatives", "prereq"),
            ("Derivatives", "Chain Rule", "prereq"),
            ("Calculus", "Chain Rule", "applies_to"),
            ("Chain Rule", "Gradient", "applies_to"),
            ("Gradient", "Backpropagation", "prereq"),
            ("Linear Algebra", "Neural Networks", "prereq"),
            ("Backpropagation", "Neural Networks", "applies_to"),
        ]

        for source, target, rel_type in relations:
            self.add_relation(source, target, rel_type)
