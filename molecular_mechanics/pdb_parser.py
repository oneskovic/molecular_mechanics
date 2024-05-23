from tempfile import NamedTemporaryFile
import os

from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import ANGSTROM2NM
from molecular_mechanics.forces import ForceField
from xyz2graph import MolGraph, to_networkx_graph
import torch

from molecular_mechanics.molecule import Graph

def _atoms_to_xyz(atoms: list[Atom]) -> str:
    xyz = f"{len(atoms)}\n\n"
    for atom in atoms:
        element = atom.element[0]
        x, y, z = atom.position / ANGSTROM2NM
        xyz += f"{element} {x} {y} {z}\n"
    return xyz

def atoms_and_bonds_from_pdb(file_path: str, forcefield : ForceField) -> tuple[list[Atom], Graph]:
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
                if atom_name == 'OXT':
                    atom_name = 'O'
                residue = tokens[3]
                molecule_id = tokens[5]
                x = float(tokens[6])
                y = float(tokens[7])
                z = float(tokens[8])
                position = torch.tensor([x, y, z])
                position *= ANGSTROM2NM
                position.requires_grad = True
                charge = residue_db.get_charge(residue, atom_name)
                element = tokens[-1]
                atom_type = [atom_type for atom_type in atom_types if atom_type.element == element][0]
                atoms.append(Atom(atom_name, charge, position, atom_type, residue=residue, molecule_number=molecule_id))

    # TODO: Connection data can be read from the pdb by looking at bonds in the residues, order of amino acids and connect records

    xyz_str = _atoms_to_xyz(atoms)
    with NamedTemporaryFile("w",delete=False) as file:
        file.write(xyz_str)

    mg = MolGraph()
    mg.read_xyz(file.name)
    os.remove(file.name)

    G = to_networkx_graph(mg)
    adjacency_list = [list(G.neighbors(n)) for n in sorted(G.nodes)]
    if sorted(G.nodes) != list(range(len(G.nodes))):
        raise ValueError("Nodes are not numbered from 0 to n-1")

    # There seems to be a bug in this library that sometimes returns an adjacency list where an atom is
    # connected to itself. We need to remove those
    for i in range(len(adjacency_list)):
        adjacency_list[i] = [j for j in adjacency_list[i] if j != i]

    return atoms, adjacency_list