import torch
from molecular_mechanics.forcefield_parser import load_forcefield
from molecular_mechanics.pdb_parser import atoms_and_bonds_from_pdb
from copy import deepcopy
from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import ANGSTROM2NM
import argparse

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

def format_pdb_line(atom_id : int, atom : Atom, molecule_id : int) -> str:
        """
        Example line:
        ATOM   2209  CB  TYR A 299       6.167  22.607  20.046  1.00  8.12           C
        00000000011111111112222222222333333333344444444445555555555666666666677777777778
        12345678901234567890123456789012345678901234567890123456789012345678901234567890

        ATOM line format description from
          http://deposit.rcsb.org/adit/docs/pdb_atom_format.html:

        COLUMNS        DATA TYPE       CONTENTS
        --------------------------------------------------------------------------------
         1 -  6        Record name     "ATOM  "
         7 - 11        Integer         Atom serial number.
        13 - 16        Atom            Atom name.
        17             Character       Alternate location indicator.
        18 - 20        Residue name    Residue name.
        22             Character       Chain identifier.
        23 - 26        Integer         Residue sequence number.
        27             AChar           Code for insertion of residues.
        31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
        39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
        47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
        55 - 60        Real(6.2)       Occupancy (Default = 1.0).
        61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
        73 - 76        LString(4)      Segment identifier, left-justified.
        77 - 78        LString(2)      Element symbol, right-justified.
        79 - 80        LString(2)      Charge on the atom.

        """
        position = atom.position.clone()
        position /= ANGSTROM2NM
        line = list(" "*80)
        line[0:6] = f"{'ATOM':<6}"
        line[6:11] = f"{atom_id:>5}"
        line[12:16] = f"{atom.element:>4}"
        line[17:20] = "HOH"
        line[21] = "A"
        line[22:26] = f"{molecule_id:>4}"
        line[30:38] = f"{position[0]:>8.3f}"
        line[38:46] = f"{position[1]:>8.3f}"
        line[46:54] = f"{position[2]:>8.3f}"
        line[54:60] = f"{1.00:>6.2f}"
        line[60:66] = f"{0.00:>6.2f}"
        line[76:78] = f"{atom.atom_type.element:>2}"
        #return f"{'ATOM':<6}{atom_id:>5} {atom.element:>4} {'HOH':>3} A{molecule_id:>4} {atom.position[0]:>8.3f}{atom.position[1]:>8.3f}{atom.position[2]:>8.3f}{1.00:>6.2f}{0.00:>6.2f}{' ':>9}{atom.atom_type.element:>2}  \n"
        return "".join(line) + "\n"

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