from xyz2graph import MolGraph, to_networkx_graph, to_plotly_figure
# Loads the xyz file called test.xyz from the data directory
# uses the xyz2graph library to convert the xyz file to a networkx graph
# this is done using a vdw radii table by the library
# Returns a list of node names (atom types) and an adjacency list in the same order
def xyz_to_graph(xyz_id):
    # Create the MolGraph object
    mg = MolGraph()
    # Read the data from the .xyz file
    mg.read_xyz(f'data/{xyz_id}.xyz')
    # Convert the molecular graph to the NetworkX graph
    G = to_networkx_graph(mg)
    atom_types = [G.nodes[n]['element'] for n in G.nodes]
    adjacency_list = [list(G.neighbors(n)) for n in G.nodes]
    return atom_types, adjacency_list

nodes, adj_list = xyz_to_graph('test')
print()
