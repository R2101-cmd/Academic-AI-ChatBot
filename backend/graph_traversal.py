def get_path(graph, start):
    path = [start]

    for node in graph.successors(start):
        path.append(node)
        for next_node in graph.successors(node):
            path.append(next_node)
            break
        break

    return path


def explain_path(path):
    for i in range(len(path) - 1):
        print(f"{path[i]} → {path[i+1]}")
