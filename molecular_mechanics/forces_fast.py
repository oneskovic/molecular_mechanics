from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import BOLTZMANN
from molecular_mechanics.forces import HarmonicBondForceParams, SoftSearch
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
        
