from molecular_mechanics.atom import Atom, AtomType, get_dihedral_angle_positions
from molecular_mechanics.constants import COULOMB, WILDCARD
from molecular_mechanics.forces import HarmonicBondForceParams, SoftSearch, HarmonicAngleForceParams, LennardJonesForceParams, DihedralForceParams
from molecular_mechanics.residue_database import ResidueDatabase
from typing import Any
import torch
from molecular_mechanics.molecule import Graph, Bond, Dihedral, Angle
import molecular_mechanics.config as conf
from copy import deepcopy

def get_key_soft_search(key: tuple, param_dict: dict[tuple, Any], soft_searcher: SoftSearch, match_tolerance = 0.8) -> tuple:
    if key not in param_dict:
        soft_key = soft_searcher.search(key)
        if soft_key is None:
            raise KeyError(f"Bond {key} not found in force field")
        return soft_key
    return key

def get_key_soft_search_dihedral(key: tuple, param_dict: dict[tuple, Any], soft_searcher: SoftSearch, match_tolerance = 0.8) -> tuple:
    if key not in param_dict:
        wildcard_key = (WILDCARD, key[1], key[2], WILDCARD)
        if wildcard_key in param_dict:
            return wildcard_key
        soft_key = soft_searcher.search(key)
        if soft_key is None:
            soft_key = soft_searcher.search(wildcard_key)
            if soft_key is None:
                return None
                #raise KeyError(f"Dihedral {key} not found in force field")
        return soft_key
    return key

class HarmonicBondForceFast:
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
        self.k_matrix = torch.zeros(max_index + 1, max_index + 1, requires_grad=False).to(conf.TORCH_DEVICE)
        self.length_matrix = torch.zeros_like(self.k_matrix, requires_grad=False).to(conf.TORCH_DEVICE)
        for bond in bonds:
            ind1, ind2 = bond
            atom1, atom2 = atoms[ind1], atoms[ind2]
            length, k = self.get_l_k(atom1, atom2)
            self.k_matrix[ind1, ind2] = k
            self.length_matrix[ind1, ind2] = length

    def get_l_k(self, atom1: Atom, atom2: Atom) -> tuple[float, float]:
        bond = get_key_soft_search((atom1.atom_type.atom_class, atom2.atom_type.atom_class), self.bond_dict, self.soft_searcher)
        length, k = self.bond_dict[bond].length, self.bond_dict[bond].k
        return length, k
    
    def get_forces(self, atom_positions: torch.Tensor) -> torch.Tensor:
        diff = atom_positions.unsqueeze(0) - atom_positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        energy = self.k_matrix * (dist - self.length_matrix) ** 2 / 2
        return energy.sum()
        
class CoulombForceFast:
    def __init__(self, all_pairs_bond_separation: list[list[int | float]], atoms: list[Atom], non_bonded_scaling_factor: float | None):
        self.non_bonded_scaling_factor = non_bonded_scaling_factor
        charge = torch.tensor([atom.charge for atom in atoms])
        self.charge_ij = charge[:, None] * charge[None, :]
        self.separation_ij = torch.tensor(all_pairs_bond_separation).to(conf.TORCH_DEVICE)
        self.charge_ij = self.charge_ij.to(conf.TORCH_DEVICE)
        self.n_atoms = len(atoms)
        self.eye = torch.eye(self.n_atoms).to(conf.TORCH_DEVICE)

    def get_forces(self, atom_positions: torch.Tensor) -> torch.Tensor:
        position_diff_ij = atom_positions.unsqueeze(1) - atom_positions.unsqueeze(0)
        r_ij_norm = position_diff_ij.norm(dim=2) + self.eye
        coulomb = torch.triu(self.charge_ij / r_ij_norm, diagonal=1)
        bonded_mask = self.separation_ij < 3
        coulomb[bonded_mask] = 0
        if self.non_bonded_scaling_factor:
            scaled_mask = self.separation_ij == 3
            coulomb[scaled_mask] *= self.non_bonded_scaling_factor
        return COULOMB * coulomb.sum()

class HarmonicAngleForceFast:
    def __init__(self, angle_dict: dict[tuple[str, str, str], HarmonicAngleForceParams], angles: list[Angle], atoms: list[Atom]):
        self.k_ijk = torch.zeros((len(atoms), len(atoms), len(atoms))).to(conf.TORCH_DEVICE)
        self.theta_ijk = torch.zeros((len(atoms), len(atoms), len(atoms))).to(conf.TORCH_DEVICE)
        soft_searcher = SoftSearch(angle_dict)
        self.n_atoms = len(atoms)
        self.eye = torch.eye(self.n_atoms).to(conf.TORCH_DEVICE)
        for i, j, k in angles:
            atom1, atom2, atom3 = atoms[i], atoms[j], atoms[k]
            curr_atoms = get_key_soft_search((atom1.atom_type.atom_class, atom2.atom_type.atom_class, atom3.atom_type.atom_class), angle_dict, soft_searcher)
            self.k_ijk[i, j, k] = angle_dict[curr_atoms].k
            self.theta_ijk[i, j, k] = angle_dict[curr_atoms].angle

    def get_forces(self, atom_positions: torch.Tensor) -> torch.Tensor:
        position_diff_ij = atom_positions.unsqueeze(1) - atom_positions.unsqueeze(0)
        position_diff_norm = (position_diff_ij.norm(dim=2) + self.eye).unsqueeze(2)
        position_diff_ij = position_diff_ij / position_diff_norm
        angles_ijk = torch.einsum('ijv,kjv->ijk', position_diff_ij, position_diff_ij)
        angles_ijk = torch.clamp(angles_ijk, -0.9999, 0.9999)
        angles_ijk = torch.acos(angles_ijk)
        return 0.5 * (self.k_ijk * (angles_ijk - self.theta_ijk) ** 2).sum()

class LennardJonesForceFast:
    def __init__(self, lj_dict: dict[str, LennardJonesForceParams], all_pairs_bond_separation: list[list[int | float]], atoms: list[Atom], non_bonded_scaling_factor: float | None):
        self.non_bonded_scaling_factor = non_bonded_scaling_factor    
        sigmas = torch.tensor([lj_dict[atom.atom_type.atom_class].sigma for atom in atoms])
        epsilons = torch.tensor([lj_dict[atom.atom_type.atom_class].epsilon for atom in atoms])
        self.sigma_ij = (sigmas[:, None] + sigmas[None, :]) / 2
        self.sigma_ij = self.sigma_ij.to(conf.TORCH_DEVICE)
        self.epsilon_ij = torch.sqrt(epsilons[:, None] * epsilons[None, :]).to(conf.TORCH_DEVICE)
        self.separation_ij = torch.tensor(all_pairs_bond_separation).to(conf.TORCH_DEVICE)
        self.eye = torch.eye(len(atoms)).to(conf.TORCH_DEVICE)

    def get_forces(self, atom_positions: torch.Tensor) -> torch.Tensor:
        position_diff_ij = atom_positions.unsqueeze(1) - atom_positions.unsqueeze(0)
        r_ij_norm = position_diff_ij.norm(dim=2) + self.eye
        lj = 4 * self.epsilon_ij * ((self.sigma_ij / r_ij_norm) ** 12 - (self.sigma_ij / r_ij_norm) ** 6)
        lj = torch.triu(lj, diagonal=1)
        bonded_mask = self.separation_ij < 3
        lj[bonded_mask] = 0
        if self.non_bonded_scaling_factor:
            scaled_mask = self.separation_ij == 3
            lj[scaled_mask] *= self.non_bonded_scaling_factor
        return lj.sum()

class DihedralForceFast:
    def __init__(self, dihedral_dict: dict[tuple, DihedralForceParams], all_dihedrals: list[Dihedral], atoms: list[Atom]):
        self.dihedral_dict = deepcopy(dihedral_dict)
        self.all_dihedrals = all_dihedrals
        self.soft_searcher = SoftSearch(dihedral_dict)
        self.atom_elements = [atom.element for atom in atoms]
        # Add reverse dihedrals
        temp_dict = dihedral_dict.copy()
        for dihedral, params in temp_dict.items():
            self.dihedral_dict[
                (dihedral[3], dihedral[2], dihedral[1], dihedral[0])
            ] = params
        
        all_ind1, all_ind2, all_ind3, all_ind4 = [], [], [], []
        all_k, all_n, all_phi = [], [], []
        for dihedral in self.all_dihedrals:
            ind1, ind2, ind3, ind4 = dihedral
            atoms = (self.atom_elements[ind1], self.atom_elements[ind2], self.atom_elements[ind3], self.atom_elements[ind4])
            new_atoms = get_key_soft_search_dihedral(atoms, self.dihedral_dict, self.soft_searcher)
            if new_atoms is None:
                if not conf.SUPRESS_WARNINGS:
                    print(f'Warning: Dihedral {atoms} not found in force field')
                all_k.append(0.0)
                all_n.append(0)
                all_phi.append(0.0)
                all_ind1.append(ind1)
                all_ind2.append(ind2)
                all_ind3.append(ind3)
                all_ind4.append(ind4)
            else:
                n_terms = len(self.dihedral_dict[new_atoms].k)
                all_k += self.dihedral_dict[new_atoms].k
                all_n += self.dihedral_dict[new_atoms].n
                all_phi += self.dihedral_dict[new_atoms].phi
                all_ind1 += [ind1] * n_terms
                all_ind2 += [ind2] * n_terms
                all_ind3 += [ind3] * n_terms
                all_ind4 += [ind4] * n_terms

        self.all_ind1 = torch.tensor(all_ind1).to(conf.TORCH_DEVICE)
        self.all_ind2 = torch.tensor(all_ind2).to(conf.TORCH_DEVICE)
        self.all_ind3 = torch.tensor(all_ind3).to(conf.TORCH_DEVICE)
        self.all_ind4 = torch.tensor(all_ind4).to(conf.TORCH_DEVICE)
        self.all_k = torch.tensor(all_k).to(conf.TORCH_DEVICE)
        self.all_n = torch.tensor(all_n).to(conf.TORCH_DEVICE)
        self.all_phi = torch.tensor(all_phi).to(conf.TORCH_DEVICE)

    def get_forces(self, atom_positions: torch.Tensor) -> torch.Tensor:
        atom1_positions = atom_positions[self.all_ind1]
        atom2_positions = atom_positions[self.all_ind2]
        atom3_positions = atom_positions[self.all_ind3]
        atom4_positions = atom_positions[self.all_ind4]

        b1 = atom2_positions - atom1_positions
        b2 = atom3_positions - atom2_positions
        b3 = atom4_positions - atom3_positions

        cross12 = torch.cross(b1, b2, dim=1)
        cross23 = torch.cross(b2, b3, dim=1)

        n1 = cross12 / torch.linalg.norm(cross12, dim=1, keepdim=True)
        n2 = cross23 / torch.linalg.norm(cross23, dim=1, keepdim=True)

        b2_normalized = b2 / torch.linalg.norm(b2, dim=1, keepdim=True)
        m1 = torch.cross(n1, b2_normalized, dim=1)

        x = torch.einsum('ij,ij->i', n1, n2)
        y = torch.einsum('ij,ij->i', m1, n2)

        angles = torch.atan2(y, x)    
        return torch.sum(self.all_k * (1 + torch.cos(self.all_n * angles - self.all_phi)))


class ForceFieldVectorized:
    def __init__(
        self,
        atom_types: list[AtomType] | None = None,
        residue_database: ResidueDatabase | None = None,
        harmonic_bond_forces: HarmonicBondForceFast | None = None,
        harmonic_angle_forces: HarmonicAngleForceFast | None = None,
        dihedral_forces: DihedralForceFast | None = None,
        lennard_jones_forces: LennardJonesForceFast | None = None,
        coulomb_forces: CoulombForceFast | None = None
    ):
        self.residue_database = residue_database
        self.atom_types = atom_types
        self.harmonic_bond_forces = harmonic_bond_forces
        self.harmonic_angle_forces = harmonic_angle_forces
        self.dihedral_forces = dihedral_forces
        self.lennard_jones_forces = lennard_jones_forces
        self.coulomb_forces = coulomb_forces
