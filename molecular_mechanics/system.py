import itertools

import torch
from torch import Tensor

from molecular_mechanics import constants
from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import BOLTZMANN
from molecular_mechanics.forces import ForceField
from molecular_mechanics.molecule import (
    Graph,
    get_all_angles,
    get_all_bonds,
    get_all_dihedrals,
    get_all_pairs_bond_separation,
)

class _PotentialEnergyCache:
    def __init__(self):
        self._atoms_positions_hash = None
        self._energy = None

    
    def _get_positions_hash(self, atoms: list[Atom]) -> int:
        position_tensor = torch.stack([atom.position for atom in atoms])
        return hash(tuple(position_tensor.flatten().tolist()))
    

    def update(self, atoms: list[Atom], energy: Tensor) -> None:
        self._atoms_positions_hash = self._get_positions_hash(atoms)
        self._energy = energy


    def get_energy(self, atoms: list[Atom]) -> Tensor | None:
        if self._atoms_positions_hash != self._get_positions_hash(atoms):
            return None
        return self._energy

class System:
    torch.manual_seed(0)
    def __init__(
        self,
        atoms: list[Atom],
        connections: Graph,
        force_field: ForceField,
        temperature: float | None = None,
    ):
        self.atoms = atoms
        self.connections = connections
        self.force_field = force_field
        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = 293.15  # Room temperature in Kelvin

        # Precomputed properties of the system
        self.bonds = get_all_bonds(connections)
        self.angles = get_all_angles(connections)
        self.dihedrals = get_all_dihedrals(connections)
        self.all_pairs_bond_separation = get_all_pairs_bond_separation(connections)

        self.velocities = self.initialize_velocities(self.temperature)

        self._potential_energy_cache = _PotentialEnergyCache()

        # Vectorization precomputations
        charge = torch.tensor([atom.charge for atom in self.atoms])
        self.charge_ij = charge[:, None] * charge[None, :]
        self.separation_ij = torch.tensor(self.all_pairs_bond_separation)
        
        if force_field.lennard_jones_forces is not None:
            lj_dict = self.force_field.lennard_jones_forces.lj_dict
            sigmas = torch.tensor([lj_dict[atom.atom_type.atom_class].sigma for atom in self.atoms])
            epsilons = torch.tensor([lj_dict[atom.atom_type.atom_class].epsilon for atom in self.atoms])
            self.sigma_ij = (sigmas[:, None] + sigmas[None, :]) / 2
            self.epsilon_ij = torch.sqrt(epsilons[:, None] * epsilons[None, :])

    def initialize_velocities(self, temperature: float) -> Tensor:
        """
        Initialize velocities for the atoms in the system
        according to the Maxwell-Boltzmann distribution.
        """

        # TODO: Verify that this is the correct way to initialize velocities
        mass = torch.tensor([atom.atom_type.mass for atom in self.atoms])
        vx = torch.normal(mean=0, std=torch.sqrt(BOLTZMANN * temperature / mass))
        vy = torch.normal(mean=0, std=torch.sqrt(BOLTZMANN * temperature / mass))
        vz = torch.normal(mean=0, std=torch.sqrt(BOLTZMANN * temperature / mass))
        return torch.stack([vx, vy, vz], dim=1)


    def get_bonds_energy(self) -> Tensor:
        harmonic_bond_forces = self.force_field.harmonic_bond_forces
        if harmonic_bond_forces is None:
            return torch.tensor(0.0)

        energy = torch.tensor(0.0)
        for bond in self.bonds:
            ind1, ind2 = bond
            atom1, atom2 = self.atoms[ind1], self.atoms[ind2]
            energy += harmonic_bond_forces.get_force(atom1, atom2)
        return energy

    def get_angles_energy(self) -> Tensor:
        harmonic_angle_forces = self.force_field.harmonic_angle_forces
        if harmonic_angle_forces is None:
            return torch.tensor(0.0)

        energy = torch.tensor(0.0)
        for angle in self.angles:
            ind1, ind2, ind3 = angle
            atom1, atom2, atom3 = self.atoms[ind1], self.atoms[ind2], self.atoms[ind3]
            energy += harmonic_angle_forces.get_force(atom1, atom2, atom3)
        return energy

    def get_dihedrals_energy(self) -> Tensor:
        dihedral_forces = self.force_field.dihedral_forces
        if dihedral_forces is None:
            return torch.tensor(0.0)

        energy = torch.tensor(0.0)
        for dihedral in self.dihedrals:
            ind1, ind2, ind3, ind4 = dihedral
            atom1, atom2, atom3, atom4 = (
                self.atoms[ind1],
                self.atoms[ind2],
                self.atoms[ind3],
                self.atoms[ind4],
            )
            energy += dihedral_forces.get_force(atom1, atom2, atom3, atom4)
        return energy
    
    def get_lennard_jones_energy(self) -> Tensor:
        lennard_jones_forces = self.force_field.lennard_jones_forces
        total_energy = torch.tensor(0.0)
        scaling_factor = self.force_field.non_bonded_scaling_factor
        for (i, atom1), (j, atom2) in itertools.combinations(enumerate(self.atoms), 2):
            bond_separation = self.all_pairs_bond_separation[i][j]
            if bond_separation < 3:
                continue
            lj = (
                lennard_jones_forces.get_force(atom1, atom2)
                if lennard_jones_forces
                else torch.tensor(0)
            )
            if scaling_factor and bond_separation == 3:
                total_energy += lj * scaling_factor
            else:
                total_energy += lj
        return total_energy
    
    def get_lennard_jones_energy_fast(self) -> Tensor:
        ''' Vectorized version of get_lennard_jones_energy '''
        position = torch.stack([atom.position for atom in self.atoms])
        position_diff_ij = position.unsqueeze(1) - position.unsqueeze(0)
        r_ij_norm = position_diff_ij.norm(dim=2) + torch.eye(len(self.atoms))
        lj = 4 * self.epsilon_ij * ((self.sigma_ij / r_ij_norm) ** 12 - (self.sigma_ij / r_ij_norm) ** 6)
        lj = torch.triu(lj, diagonal=1)
        bonded_mask = self.separation_ij < 3
        lj[bonded_mask] = 0
        if self.force_field.non_bonded_scaling_factor:
            scaled_mask = self.separation_ij == 3
            lj[scaled_mask] *= self.force_field.non_bonded_scaling_factor
        return lj.sum()
    
    def get_coulomb_energy(self) -> Tensor:
        coulomb_forces = self.force_field.coulomb_forces
        total_energy = torch.tensor(0.0)
        scaling_factor = self.force_field.non_bonded_scaling_factor
        for (i, atom1), (j, atom2) in itertools.combinations(enumerate(self.atoms), 2):
            bond_separation = self.all_pairs_bond_separation[i][j]
            if bond_separation < 3:
                continue
            coulomb = (
                coulomb_forces.get_force(atom1, atom2)
                if coulomb_forces
                else torch.tensor(0)
            )
            if scaling_factor and bond_separation == 3:
                total_energy += coulomb * scaling_factor
            else:
                total_energy += coulomb
        return total_energy
    
    def get_coulomb_energy_fast(self) -> Tensor:
        ''' Vectorized version of get_coulomb_energy '''
        position = torch.stack([atom.position for atom in self.atoms])
        position_diff_ij = position.unsqueeze(1) - position.unsqueeze(0)
        r_ij_norm = position_diff_ij.norm(dim=2) + torch.eye(len(self.atoms))
        coulomb = torch.triu(self.charge_ij / r_ij_norm, diagonal=1)
        bonded_mask = self.separation_ij < 3
        coulomb[bonded_mask] = 0
        if self.force_field.non_bonded_scaling_factor:
            scaled_mask = self.separation_ij == 3
            coulomb[scaled_mask] *= self.force_field.non_bonded_scaling_factor
        return constants.COULOMB * coulomb.sum()


    def get_non_bonded_energy(self) -> Tensor:
        energy = torch.tensor(0.0)
        if self.force_field.lennard_jones_forces:
            energy += self.get_lennard_jones_energy_fast()
        if self.force_field.coulomb_forces:
            energy += self.get_coulomb_energy_fast()
        return energy
    

    def get_potential_energy(self) -> Tensor:
        cached_energy = self._potential_energy_cache.get_energy(self.atoms)
        if cached_energy is not None:
            return cached_energy
        
        total_energy = torch.tensor(0.0)
        total_energy += self.get_bonds_energy()
        total_energy += self.get_angles_energy()
        total_energy += self.get_dihedrals_energy()
        total_energy += self.get_non_bonded_energy()
        self._potential_energy_cache.update(self.atoms, total_energy)
        return total_energy
    
    def get_kinetic_energy(self) -> Tensor:
        mass = torch.tensor([atom.atom_type.mass for atom in self.atoms])
        velocity_norms = self.velocities.norm(dim=1)
        return 0.5 * (mass * velocity_norms ** 2).sum()
