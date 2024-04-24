import itertools

# Adjacency list of atoms that are connected by a bond
# Each atom is represented by an index in the list
type Graph = list[list[int]]

type Bond = tuple[int, int]
type Angle = tuple[int, int, int]
type Dihedral = tuple[int, int, int, int]


def bfs(source: int, graph: Graph) -> list[int | float]:
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


def get_all_pairs_bond_separation(graph: Graph) -> list[list[int | float]]:
    """
    Returns a matrix of distances between all pairs of atoms in the graph.
    If two atoms are not connected, the distance is infinity.
    """
    all_pairs_bond_separation = []
    for atom in range(len(graph)):
        distance_from_atom = bfs(atom, graph)
        all_pairs_bond_separation.append(distance_from_atom)
    return all_pairs_bond_separation


def get_all_bonds(graph: Graph) -> list[Bond]:
    """
    Returns a list of all bonds in the graph.
    Bonds are not repeated, i.e. if bond (i, j) is in the list, (j, i) is not.
    """
    all_bonds = []
    for atom, neighbors in enumerate(graph):
        for neighbor in neighbors:
            if atom < neighbor:
                all_bonds.append((atom, neighbor))
    return all_bonds


def get_all_angles(graph: Graph) -> list[Angle]:
    """
    Returns a list of all angles in the graph.
    Angles are not repeated, i.e. if angle (i, j, k) is in the list, (k, j, i) is not.
    """
    all_angles = []
    # Try to fix each atom as the middle atom
    for atom2, neighbors in enumerate(graph):
        # Loop over all pairs of its neighbors
        for atom1, atom3 in itertools.combinations(neighbors, 2):
            all_angles.append((atom1, atom2, atom3))
    return all_angles


def get_all_dihedrals(graph: Graph) -> list[Dihedral]:
    """
    Returns a list of all dihedrals in the graph.
    Dihedrals are not repeated, i.e. if dihedral (i, j, k, l) is in the list, (l, k, j, i) is not.
    """
    all_dihedrals = []
    # Try to fix two atoms as the middle atoms
    for atom2, neighbors in enumerate(graph):
        for atom3 in neighbors:
            # Ensure index of atom2 < atom3 to avoid double counting
            if atom2 >= atom3:
                continue
            # Form dihedrals by considering (atom1, atom4)
            # where atom1 and atom4 are neighbors of atom2 and atom3 respectively
            for atom1, atom4 in itertools.product(graph[atom2], graph[atom3]):
                # atom2 and atom3 are each other's neighbors
                # so we skip over this case
                if atom1 == atom3 or atom4 == atom2:
                    continue
                # Skip over "ring" configurations i.e. A1-A2-A3-A1
                if atom1 == atom4:
                    continue
                all_dihedrals.append((atom1, atom2, atom3, atom4))

    return all_dihedrals
