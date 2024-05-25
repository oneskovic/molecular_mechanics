from molecular_mechanics.atom import get_bond_angle, get_dihedral_angle, get_distance
from molecular_mechanics.constants import ANGSTROM2NM
from molecular_mechanics.system import System
from molecular_mechanics.system_fast import SystemFast
from molecular_mechanics.atom import Atom
import torch
from utilities.pdb_util import format_pdb_line

def print_system_state(system: System | SystemFast, bonds=False, angles=False, dihedrals=False) -> None:
    description_width = 25
    value_width = 20
    precision = 6

    header = "STATE"
    width = description_width + value_width
    dashes = "-" * ((width - len(header)) // 2)
    print(dashes + header + dashes)
    desc_val_dict: dict[str, float] = {}

    if isinstance(system, SystemFast):
        potential_energy = system.get_potential_energy()
        kinetic_energy = system.get_kinetic_energy()
        with torch.no_grad():
            desc_val_dict["Potential energy"] = potential_energy.item()
            desc_val_dict["Kinetic energy"] = kinetic_energy.item()
            desc_val_dict["Total energy"] = (potential_energy + kinetic_energy).item()

    if isinstance(system, System):
        potential_energy = system.get_potential_energy()
        kinetic_energy = system.get_kinetic_energy()
        desc_val_dict["Potential energy"] = potential_energy.item()
        desc_val_dict["Kinetic energy"] = kinetic_energy.item()
        desc_val_dict["Total energy"] = (potential_energy + kinetic_energy).item()

        if bonds:
            # FIXME: Duplicate code from get_bonds_energy
            for bond in system.bonds:
                ind1, ind2 = bond
                atom1, atom2 = system.atoms[ind1], system.atoms[ind2]
                description = f"Bond length {atom1.element}{ind1}-{atom2.element}{ind2}:"
                distance = get_distance(atom1, atom2)
                desc_val_dict[description] = distance.item()
        if angles:
            # FIXME: Duplicate code from get_angles_energy
            for angle in system.angles:
                i, j, k = angle
                atom1, atom2, atom3 = system.atoms[i], system.atoms[j], system.atoms[k]
                atom1_str = f"{atom1.element}{i}"
                atom2_str = f"{atom2.element}{j}"
                atom3_str = f"{atom3.element}{k}"
                description = f"Angle {atom1_str}-{atom2_str}-{atom3_str}:"
                angle_val = get_bond_angle(atom1, atom2, atom3)
                desc_val_dict[description] = angle_val.item()

        if dihedrals:
            for dihedral in system.dihedrals:
                i, j, k, m = dihedral
                atom1, atom2, atom3, atom4 = (
                    system.atoms[i],
                    system.atoms[j],
                    system.atoms[k],
                    system.atoms[m],
                )
                atom1_str = f"{atom1.element}{i}"
                atom2_str = f"{atom2.element}{j}"
                atom3_str = f"{atom3.element}{k}"
                atom4_str = f"{atom4.element}{m}"
                description = f"Dihedral {atom1_str}-{atom2_str}-{atom3_str}-{atom4_str}:"
                angle_val = get_dihedral_angle(atom1, atom2, atom3, atom4)
                desc_val_dict[description] = angle_val.item()

    for description, value in desc_val_dict.items():
        print(f"{description:<{description_width}}{value:>{value_width}.{precision}f}")
    print("-" * width)


class XYZTrajectoryWriter:
    def __init__(self, filename: str) -> None:
        self._file = open(filename, "w")
        
    def write(self, system: System | SystemFast) -> None:
        if isinstance(system, System):
            self._file.write(f"{len(system.atoms)}\n")  # number of atoms
            self._file.write("\n")  # name of molecule
            for atom in system.atoms:
                x, y, z = [coord / ANGSTROM2NM for coord in atom.position.tolist()]
                self._file.write(f"{atom.element} {x:6f} {y:6f} {z:6f}\n")
        elif isinstance(system, SystemFast):
            n = system.atom_positions.shape[0]
            self._file.write(f"{n}\n")  # number of atoms
            self._file.write("\n")  # name of molecule
            for i in range(n):
                x, y, z = [coord / ANGSTROM2NM for coord in system.atom_positions[i].tolist()]
                element = system.atom_elements[i]
                self._file.write(f"{element} {x:6f} {y:6f} {z:6f}\n")
    
    def close(self) -> None:
        self._file.close()

class PDBTrajectoryWriter:
    def __init__(self, filename: str) -> None:
        self._file = open(filename, "w")
        header = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1 "
        self._file.write(header + "\n")
        self.model_number = 1
    
    def write(self, system: System | SystemFast) -> None:
        if isinstance(system, System):
            pass
        else:
            model_line = f"MODEL        {self.model_number}"
            self._file.write(model_line + "\n")
            self.model_number += 1
            for i in range(len(system.atom_elements)):
                x,y,z = [coord for coord in system.atom_positions[i].tolist()]
                atom = Atom(system.atom_elements[i], system.atom_charges[i], torch.tensor([x,y,z]), system.atom_types[i], system.atom_residues[i], system.atom_molecule_numbers[i])
                line = format_pdb_line(i+1, atom, atom.molecule_number)
                self._file.write(line)

            line = list(" "*80)
            last_molecule_id = system.atom_molecule_numbers[-1]
            last_atom_id = len(system.atom_elements)
            last_residue = system.atom_residues[-1]
            atom_element = ""
            line = list(" "*80)
            line[0:6] = f"{'TER':<6}"
            line[6:11] = f"{(last_atom_id+1):>5}"
            line[12:16] = f"{atom_element:>4}"
            line[17:20] = f"{last_residue:>3}"
            line[21] = "A"
            line[22:26] = f"{last_molecule_id:>4}"
            self._file.write("".join(line) + "\n")
            self._file.write("ENDMDL\n")
    def close(self) -> None:
        self._file.close()