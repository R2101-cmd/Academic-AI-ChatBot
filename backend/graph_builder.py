import networkx as nx


def build_graph():
    G = nx.DiGraph()

    G.add_edge("Algebra", "Calculus")
    G.add_edge("Calculus", "Gradient Descent")
    G.add_edge("Gradient Descent", "Backpropagation")

    return G
