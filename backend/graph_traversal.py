def get_path(graph, start, max_hops=6):
    path = []
    visited = set()
    current = start

    for _ in range(max_hops + 1):
        if not current or current in visited:
            break
        path.append(current)
        visited.add(current)
        successors = [node for node in graph.successors(current) if node not in visited]
        if not successors:
            break
        current = successors[0]

    return path


def explain_path(path):
    for i in range(len(path) - 1):
        print(f"{path[i]} -> {path[i + 1]}")
