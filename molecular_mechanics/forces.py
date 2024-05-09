import typing
import torch

from molecular_mechanics.atom import (
    Atom,
    get_bond_angle,
    get_dihedral_angle,
    get_distance,
)
from molecular_mechanics.constants import COULOMB
from molecular_mechanics.residue_database import ResidueDatabase
from molecular_mechanics.atom import AtomType

class HarmonicBondForceParams:
    def __init__(self, length: float, k: float):
        self.length = length
        self.k = k


class HarmonicBondForce:
    def __init__(self, bond_dict: dict[tuple, HarmonicBondForceParams]):
        self.bond_dict = bond_dict
        # Add reverse bonds
        temp_dict = bond_dict.copy()
        for bond, params in temp_dict.items():
            self.bond_dict[(bond[1], bond[0])] = params

    def get_force(self, atom1: Atom, atom2: Atom) -> torch.Tensor:
        bond = (atom1.atom_type.atom_class, atom2.atom_type.atom_class)
        force = torch.tensor(0.0)
        if bond in self.bond_dict:
            length, k = self.bond_dict[bond].length, self.bond_dict[bond].k
            dist = (atom1.position - atom2.position).norm()
            k /= 1000.0
            length *= 10.0
            force = k * (dist - length) ** 2 / 2
        return force


class HarmonicAngleForceParams:
    def __init__(self, angle: float, k: float):
        self.angle = angle
        self.k = k


class HarmonicAngleForce:
    def __init__(self, angle_dict: dict[tuple, HarmonicAngleForceParams]):
        self.angle_dict = angle_dict
        # Add reverse angles
        temp_dict = angle_dict.copy()
        for angle, params in temp_dict.items():
            self.angle_dict[(angle[2], angle[1], angle[0])] = params

    def get_force(self, atom1: Atom, atom2: Atom, atom3: Atom) -> torch.Tensor:
        atoms = (atom1.atom_type.atom_class, atom2.atom_type.atom_class, atom3.atom_type.atom_class)
        if atoms not in self.angle_dict:
            raise KeyError(f"Angle {atoms} not found in force field")

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
            raise KeyError(f"Dihedral {atoms} not found in force field")

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
        if atom1.element not in self.lj_dict:
            raise KeyError(f"Lennard-Jones parameters not found for {atom1.element}")
        if atom2.element not in self.lj_dict:
            raise KeyError(f"Lennard-Jones parameters not found for {atom2.element}")

        epsilon1, sigma1 = (
            self.lj_dict[atom1.element].epsilon,
            self.lj_dict[atom1.element].sigma,
        )
        epsilon2, sigma2 = (
            self.lj_dict[atom2.element].epsilon,
            self.lj_dict[atom2.element].sigma,
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
        residue_database: ResidueDatabase,
        atom_types: list[AtomType],
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
