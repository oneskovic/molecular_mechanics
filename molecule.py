def bfs(source, graph):
    # Returns a list of distances from the source to all other nodes
    # in the graph
    queue = [source]
    visited = set()
    distance = [float("inf")] * len(graph)
    distance[source] = 0
    while queue:
        node = queue.pop(0)
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                distance[neighbor] = distance[node] + 1
    return distance


def get_all_pairs_bond_separation(graph):
    # Returns a dictionary of all pairs of atoms and their bond separation
    # in the graph
    all_pairs_bond_separation = []
    for atom in range(len(graph)):
        distance_from_atom = bfs(atom, graph)
        all_pairs_bond_separation.append(distance_from_atom)
    return all_pairs_bond_separation


def get_all_bonds(graph):
    # Returns a list of all bonds in the graph
    all_bonds = []
    for atom, neighbors in enumerate(graph):
        for neighbor in neighbors:
            all_bonds.append((atom, neighbor))
    return all_bonds
