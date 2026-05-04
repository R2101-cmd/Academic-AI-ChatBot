"""Graph-CoT learning path construction."""

from __future__ import annotations

import networkx as nx


class GraphCoTModule:
    """Build a concept graph and return prerequisite-aware learning paths."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        edges = [
            ("Algebra", "Calculus", "prerequisite"),
            ("Calculus", "Derivatives", "prerequisite"),
            ("Derivatives", "Chain Rule", "prerequisite"),
            ("Chain Rule", "Gradient Descent", "supports"),
            ("Gradient Descent", "Backpropagation", "supports"),
            ("Linear Algebra", "Neural Networks", "prerequisite"),
            ("Neural Networks", "Backpropagation", "application"),
            ("Programming", "Algorithms", "prerequisite"),
            ("Algorithms", "Machine Learning", "supports"),
            ("Statistics", "Machine Learning", "prerequisite"),
            ("Machine Learning", "Artificial Intelligence", "part_of"),
            ("Research Methods", "Academic Writing", "supports"),
        ]
        for source, target, relation in edges:
            self.graph.add_edge(source, target, relation=relation)

    def build_path(self, topic: str, concepts: list[str] | None = None) -> list[str]:
        lowered = topic.lower()
        keyword_targets = {
            "backprop": "Backpropagation",
            "gradient": "Gradient Descent",
            "chain rule": "Chain Rule",
            "neural": "Neural Networks",
            "machine learning": "Machine Learning",
            "artificial intelligence": "Artificial Intelligence",
            "algorithm": "Algorithms",
            "calculus": "Calculus",
            "writing": "Academic Writing",
        }

        target = next((value for key, value in keyword_targets.items() if key in lowered), None)
        if not target and concepts:
            target = next((node for node in self.graph.nodes if node.lower() in " ".join(concepts).lower()), None)
        if not target:
            return ["Foundations", "Core Concept", topic.title(), "Practice", "Revision"]

        candidates = []
        for source in self.graph.nodes:
            try:
                path = nx.shortest_path(self.graph, source, target)
                if len(path) > 1:
                    candidates.append(path)
            except nx.NetworkXNoPath:
                continue
        return max(candidates, key=len) if candidates else [target]

    def graph_payload(self) -> dict[str, list[dict[str, str]]]:
        return {
            "nodes": [{"id": node, "label": node} for node in self.graph.nodes],
            "edges": [
                {"source": source, "target": target, "relation": data["relation"]}
                for source, target, data in self.graph.edges(data=True)
            ],
        }

