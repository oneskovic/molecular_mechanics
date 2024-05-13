from difflib import SequenceMatcher
from functools import cache
from typing import Any
import typing

import torch

from molecular_mechanics.atom import Atom, AtomType, get_bond_angle, get_dihedral_angle, get_distance
from molecular_mechanics.constants import COULOMB
from molecular_mechanics.residue_database import ResidueDatabase

class SoftSearch:
    def __init__(self, param_dict: dict[tuple, Any]):
        self.param_dict = param_dict
    
    @cache
    def search(self, key: tuple[str, ...], match_tolerance = 0.8) -> tuple[str,str,str] | None:
        atoms_str = "".join(key)
        best_match = None
        best_match_factor = 0
        for dict_atoms in self.param_dict:
            # Concatenate the atoms in the dictionary
            dict_atoms_str = "".join(dict_atoms)
            # Remove numbers and * from the string
            dict_atoms_str = "".join(filter(str.isalpha, dict_atoms_str))

            match_result = SequenceMatcher(None, atoms_str, dict_atoms_str).ratio()
            if match_result > best_match_factor:
                best_match = dict_atoms
                best_match_factor = match_result

        if best_match_factor < match_tolerance:
            return None
        return best_match


class HarmonicBondForceParams:
    def __init__(self, length: float, k: float):
        self.length = length
        self.k = k


class HarmonicBondForce:
    def __init__(self, bond_dict: dict[tuple, HarmonicBondForceParams]):
        self.bond_dict = bond_dict
        self.soft_searcher = SoftSearch(bond_dict)
        # Add reverse bonds
        temp_dict = bond_dict.copy()
        for bond, params in temp_dict.items():
            self.bond_dict[(bond[1], bond[0])] = params
    
    def get_force(self, atom1: Atom, atom2: Atom) -> torch.Tensor:
        bond = (atom1.atom_type.atom_class, atom2.atom_type.atom_class)
        if bond not in self.bond_dict:
            soft_key = self.soft_searcher.search(bond)
            if soft_key is None:
                raise KeyError(f"Bond {bond} not found in force field")
            bond = soft_key
        force = torch.tensor(0.0)
        length, k = self.bond_dict[bond].length, self.bond_dict[bond].k
        dist = (atom1.position - atom2.position).norm()
        force = k * (dist - length) ** 2 / 2
        return force


class HarmonicAngleForceParams:
    def __init__(self, angle: float, k: float):
        self.angle = angle
        self.k = k


class HarmonicAngleForce:
    def __init__(self, angle_dict: dict[tuple[str,str,str], HarmonicAngleForceParams]):
        self.angle_dict = angle_dict
        self.soft_searcher = SoftSearch(angle_dict)
        # Add reverse angles
        temp_dict = angle_dict.copy()
        for angle, params in temp_dict.items():
            self.angle_dict[(angle[2], angle[1], angle[0])] = params

    def get_force(self, atom1: Atom, atom2: Atom, atom3: Atom) -> torch.Tensor:
        atoms = (atom1.atom_type.atom_class, atom2.atom_type.atom_class, atom3.atom_type.atom_class)
        if atoms not in self.angle_dict:
            soft_key = self.soft_searcher.search(atoms) 
            if soft_key is None:
                raise KeyError(f"Angle {atoms} not found in force field")
            atoms = soft_key

        angle, k = self.angle_dict[atoms].angle, self.angle_dict[atoms].k
        current_angle = get_bond_angle(atom1, atom2, atom3)
        force = k * (current_angle - angle) ** 2 / 2
        return force


class DihedralForceParams:
    def __init__(self, k: float, n: int, phi: float):
        self.k = k
        self.n = n
        self.phi = phi


class DihedralForce:
    def __init__(self, dihedral_dict: dict[tuple, DihedralForceParams]):
        self.dihedral_dict = dihedral_dict
        self.soft_searcher = SoftSearch(dihedral_dict)
        # Add reverse dihedrals
        temp_dict = dihedral_dict.copy()
        for dihedral, params in temp_dict.items():
            self.dihedral_dict[
                (dihedral[3], dihedral[2], dihedral[1], dihedral[0])
            ] = params

    def get_force(
        self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom
    ) -> torch.Tensor:
        atoms = (atom1.element, atom2.element, atom3.element, atom4.element)
        if atoms not in self.dihedral_dict:
            soft_key = self.soft_searcher.search(atoms)
            if soft_key is None:
                raise KeyError(f"Dihedral {atoms} not found in force field")
            atoms = soft_key

        k, n, phi = (
            self.dihedral_dict[atoms].k,
            self.dihedral_dict[atoms].n,
            self.dihedral_dict[atoms].phi,
        )
        angle = get_dihedral_angle(atom1, atom2, atom3, atom4)
        force = k * (1 + torch.cos(n * angle - phi))
        return force


class LennardJonesForceParams:
    def __init__(self, epsilon: float, sigma: float):
        assert epsilon >= 0
        assert sigma >= 0
        self.epsilon = epsilon
        self.sigma = sigma


class LennardJonesForce:
    def __init__(self, lj_dict: dict[str, LennardJonesForceParams]):
        self.lj_dict = lj_dict

    def get_force(self, atom1: Atom, atom2: Atom) -> torch.Tensor:
        class1 = atom1.atom_type.atom_class
        class2 = atom2.atom_type.atom_class
        if class1 not in self.lj_dict:
            raise KeyError(f"Lennard-Jones parameters not found for {class1}")
        if class2 not in self.lj_dict:
            raise KeyError(f"Lennard-Jones parameters not found for {class2}")

        epsilon1, sigma1 = (
            self.lj_dict[class1].epsilon,
            self.lj_dict[class1].sigma,
        )
        epsilon2, sigma2 = (
            self.lj_dict[class2].epsilon,
            self.lj_dict[class2].sigma,
        )
        dist = (atom1.position - atom2.position).norm()
        dist = typing.cast(torch.Tensor, dist)
        sigma = (sigma1 + sigma2) / 2
        epsilon = (epsilon1 * epsilon2) ** 0.5
        epsilon = typing.cast(float, epsilon)

        force = 4 * epsilon * ((sigma / dist) ** 12 - (sigma / dist) ** 6)
        return force


class CoulombForce:
    def __init__(self):
        pass

    def get_force(self, atom1: Atom, atom2: Atom) -> torch.Tensor:
        charge1, charge2 = (
            atom1.charge,
            atom2.charge,
        )
        dist = get_distance(atom1, atom2)
        force = COULOMB * charge1 * charge2 / dist
        return force


class ForceField:
    def __init__(
        self,
        atom_types: list[AtomType] | None = None,
        residue_database: ResidueDatabase | None = None,
        harmonic_bond_forces: HarmonicBondForce | None = None,
        harmonic_angle_forces: HarmonicAngleForce | None = None,
        dihedral_forces: DihedralForce | None = None,
        lennard_jones_forces: LennardJonesForce | None = None,
        coulomb_forces: CoulombForce | None = None,
        non_bonded_scaling_factor: float | None = None,
    ):
        self.residue_database = residue_database
        self.atom_types = atom_types
        self.harmonic_bond_forces = harmonic_bond_forces
        self.harmonic_angle_forces = harmonic_angle_forces
        self.dihedral_forces = dihedral_forces
        self.lennard_jones_forces = lennard_jones_forces
        self.coulomb_forces = coulomb_forces
        self.non_bonded_scaling_factor = non_bonded_scaling_factor
