import torch
from molecular_mechanics.forcefield_parser import load_forcefield
from molecular_mechanics.pdb_parser import atoms_and_bonds_from_pdb
from copy import deepcopy
from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import ANGSTROM2NM
import argparse
from utilities.pdb_util import format_pdb_line

def get_bounding_box(atoms):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    for atom in atoms:
        x, y, z = atom.position
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def atoms_to_pdb(molecules : list[list[Atom]], filename):
    with open(filename, 'w+') as f:
        molecule_id = 1
        atom_id = 1
        for molecule in molecules:
            for atom in molecule:
                f.write(format_pdb_line(atom_id, atom, molecule_id))
                atom_id += 1
            molecule_id += 1
    
force_field = load_forcefield('data/tip3p.xml')
atoms, connections = atoms_and_bonds_from_pdb(str('data/water/water.pdb'), force_field)
oxygen_atom = [atom for atom in atoms if atom.atom_type.element == 'O'][0]
hydrogen_atoms = [atom for atom in atoms if atom.atom_type.element == 'H']
min_coords, max_coords = get_bounding_box(atoms)
side_lengths = [max_coords[i] - min_coords[i] for i in range(3)]
padding = 0.15

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int)
parser.add_argument("m", type=int)
parser.add_argument("l", type=int)
args = parser.parse_args()
n, m, l = args.n, args.m, args.l

new_water_molecules = []
with torch.no_grad():
    for i in range(n):
        for j in range(m):
            for k in range(l):
                x_offset = i * (side_lengths[0] + padding)
                y_offset = j * (side_lengths[1] + padding)
                z_offset = k * (side_lengths[2] + padding)
                new_molecule = []
                for atom in atoms:
                    new_atom = deepcopy(atom)
                    new_atom.position[0] += x_offset
                    new_atom.position[1] += y_offset
                    new_atom.position[2] += z_offset
                    new_molecule.append(new_atom)
                new_water_molecules.append(new_molecule)

atoms_to_pdb(new_water_molecules, f'water_{n}_{m}_{l}.pdb')