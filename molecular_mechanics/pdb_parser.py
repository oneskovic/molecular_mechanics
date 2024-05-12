from molecular_mechanics.atom import Atom
from molecular_mechanics.forces import ForceField
from xyz2graph import MolGraph, to_networkx_graph
import torch

def __atoms_to_xyz(atoms: list[Atom]) -> str:
    xyz = f"{len(atoms)}\n\n"
    for atom in atoms:
        element = atom.element[0] # First character of the atom name is the element
        xyz += f"{element} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
    return xyz

def atoms_and_bonds_from_pdb(file_path: str, forcefield : ForceField) -> tuple[list[Atom], list[list[int]]]:
    residue_db = forcefield.residue_database
    if residue_db is None:
        raise ValueError("Forcefield does not have a residue database")
    atom_types = forcefield.atom_types
    if atom_types is None:
        raise ValueError("Forcefield does not have atom types")
    # Read the atoms from the pdb file
    atoms = []
    with open(file_path, "r") as file:
        atoms = []
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                tokens = line.split()
                atom_name = tokens[2]
                residue = tokens[3]
                position = torch.tensor([float(tokens[6]), float(tokens[7]), float(tokens[8])], requires_grad=True)
                charge = residue_db.get_charge(residue, atom_name)
                
                element = tokens[-1]
                atom_type = [atom_type for atom_type in atom_types if atom_type.element == element][0]
                atoms.append(Atom(atom_name, charge, position, atom_type))

    # TODO: Connection data can be read from the pdb by looking at bonds in the residues, order of amino acids and connect records

    # Write the atoms to a temporary .xyz file
    xyz_str = __atoms_to_xyz(atoms)
    with open("data/tmp.xyz", "w") as file:
        file.write(xyz_str)
    
    mg = MolGraph()
    # Read the data from the .xyz file
    mg.read_xyz(f'data/tmp.xyz')
    # Convert the molecular graph to the NetworkX graph
    G = to_networkx_graph(mg)
    adjacency_list = [list(G.neighbors(n)) for n in sorted(G.nodes)]

    # There seems to be a bug in this library that sometimes returns an adjacency list where an atom is
    # connected to itself. We need to remove those
    for i in range(len(adjacency_list)):
        adjacency_list[i] = [j for j in adjacency_list[i] if j != i]

    return atoms, adjacency_list