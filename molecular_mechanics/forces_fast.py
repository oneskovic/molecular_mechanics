from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import BOLTZMANN, COULOMB
from molecular_mechanics.forces import HarmonicBondForceParams, SoftSearch, HarmonicAngleForceParams, LennardJonesForceParams
from molecular_mechanics.system import System
import torch


class HarmonicBondForce:
    bonded_mask : torch.Tensor
    k_matrix : torch.Tensor
    length_matrix : torch.Tensor
    def __init__(self, bond_dict: dict[tuple, HarmonicBondForceParams], bonds: list[tuple[int, int]], atoms: list[Atom]):
        self.bond_dict = bond_dict
        self.soft_searcher = SoftSearch(bond_dict)
        # Add reverse bonds
        temp_dict = bond_dict.copy()
        for bond, params in temp_dict.items():
            self.bond_dict[(bond[1], bond[0])] = params
        
        # Create bonded mask
        max_index = max([max(bond) for bond in bonds])
        self.k_matrix = torch.zeros(max_index + 1, max_index + 1, requires_grad=False)
        self.length_matrix = torch.zeros_like(self.k_matrix, requires_grad=False)
        for bond in bonds:
            ind1, ind2 = bond
            atom1, atom2 = atoms[ind1], atoms[ind2]
            length, k = self.get_l_k(atom1, atom2)
            self.k_matrix[ind1, ind2] = k
            self.length_matrix[ind1, ind2] = length

    def get_l_k(self, atom1: Atom, atom2: Atom) -> tuple[float, float]:
        bond = (atom1.atom_type.atom_class, atom2.atom_type.atom_class)
        if bond not in self.bond_dict:
            soft_key = self.soft_searcher.search(bond)
            if soft_key is None:
                raise KeyError(f"Bond {bond} not found in force field")
            bond = soft_key
        length, k = self.bond_dict[bond].length, self.bond_dict[bond].k
        return length, k
    
    def get_forces(self, atoms: list[Atom]) -> torch.Tensor:
        atom_positions = torch.stack([atom.position for atom in atoms])
        diff = atom_positions.unsqueeze(0) - atom_positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        energy = self.k_matrix * (dist - self.length_matrix) ** 2 / 2
        return energy.sum()
        
class CoulombForceFast:
    def __init__(self, all_pairs_bond_separation: list[list[int | float]], atoms: list[Atom]):
        charge = torch.tensor([atom.charge for atom in atoms])
        self.charge_ij = charge[:, None] * charge[None, :]
        self.separation_ij = torch.tensor(all_pairs_bond_separation)

    def get_forces(self, atoms: list[Atom]) -> torch.Tensor:
        position = torch.stack([atom.position for atom in atoms])
        position_diff_ij = position.unsqueeze(1) - position.unsqueeze(0)
        r_ij_norm = position_diff_ij.norm(dim=2) + torch.eye(len(atoms))
        coulomb = torch.triu(self.charge_ij / r_ij_norm, diagonal=1)
        bonded_mask = self.separation_ij < 3
        coulomb[bonded_mask] = 0
        if self.force_field.non_bonded_scaling_factor:
            scaled_mask = self.separation_ij == 3
            coulomb[scaled_mask] *= self.force_field.non_bonded_scaling_factor
        return COULOMB * coulomb.sum()

class HarmonicAngleForceFast:
    def __init__(self, angle_dict: dict[tuple[str, str, str], HarmonicAngleForceParams], atoms: list[Atom]):
        self.k_ijk = torch.zeros((len(atoms), len(atoms), len(atoms)))
        self.theta_ijk = torch.zeros((len(atoms), len(atoms), len(atoms)))
        for i, j, k in self.angles:
            atom1, atom2, atom3 = atoms[i], atoms[j], atoms[k]
            curr_atoms = (atom1.atom_type.atom_class, atom2.atom_type.atom_class, atom3.atom_type.atom_class)
            self.k_ijk[i, j, k] = angle_dict[curr_atoms].k
            self.theta_ijk[i, j, k] = angle_dict[curr_atoms].angle

    def get_forces(self, atoms: list[Atom]) -> torch.Tensor:
        position = torch.stack([atom.position for atom in atoms])
        position_diff_ij = position.unsqueeze(1) - position.unsqueeze(0)
        position_diff_norm = (position_diff_ij.norm(dim=2) + torch.eye(len(atoms))).unsqueeze(2)
        position_diff_ij = position_diff_ij / position_diff_norm
        angles_ijk = torch.einsum('ijv,kjv->ijk', position_diff_ij, position_diff_ij)
        angles_ijk = torch.clamp(angles_ijk, -0.9999, 0.9999)
        angles_ijk = torch.acos(angles_ijk)
        return 0.5 * (self.k_ijk * (angles_ijk - self.theta_ijk) ** 2).sum()

class LennardJonesForceFast:
    def __init__(self, lj_dict: dict[str, LennardJonesForceParams], all_pairs_bond_separation: list[list[int | float]], atoms: list[Atom], non_bonded_scaling_factor: float):
        self.non_bonded_scaling_factor = non_bonded_scaling_factor
        sigmas = torch.tensor([lj_dict[atom.atom_type.atom_class].sigma for atom in atoms])
        epsilons = torch.tensor([lj_dict[atom.atom_type.atom_class].epsilon for atom in atoms])
        self.sigma_ij = (sigmas[:, None] + sigmas[None, :]) / 2
        self.epsilon_ij = torch.sqrt(epsilons[:, None] * epsilons[None, :])
        self.separation_ij = torch.tensor(all_pairs_bond_separation)

    def get_forces(self, atoms: list[Atom]) -> torch.Tensor:
        position = torch.stack([atom.position for atom in atoms])
        position_diff_ij = position.unsqueeze(1) - position.unsqueeze(0)
        r_ij_norm = position_diff_ij.norm(dim=2) + torch.eye(len(atoms))
        lj = 4 * self.epsilon_ij * ((self.sigma_ij / r_ij_norm) ** 12 - (self.sigma_ij / r_ij_norm) ** 6)
        lj = torch.triu(lj, diagonal=1)
        bonded_mask = self.separation_ij < 3
        lj[bonded_mask] = 0
        if self.non_bonded_scaling_factor:
            scaled_mask = self.separation_ij == 3
            lj[scaled_mask] *= self.non_bonded_scaling_factor
        return lj.sum()
