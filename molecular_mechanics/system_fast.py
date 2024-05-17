import itertools

import torch
from torch import Tensor

from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import BOLTZMANN
from molecular_mechanics.forces_fast import ForceFieldVectorized
from molecular_mechanics.molecule import (
    Graph,
    get_all_angles,
    get_all_bonds,
    get_all_dihedrals,
    get_all_pairs_bond_separation,
)
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _PotentialEnergyCache:
    def __init__(self):
        self._atoms_positions_hash = None
        self._energy = None

    def _get_positons_hash(self, atom_positions: Tensor) -> int:
        return hash(tuple(atom_positions.flatten().tolist()))
    
    def update(self, atom_positions: Tensor, energy: Tensor) -> None:
        self._atoms_positions_hash = self._get_positons_hash(atom_positions)
        self._energy = energy
    
    def get_energy(self, atom_positions: Tensor) -> Tensor | None:
        if self._atoms_positions_hash != self._get_positons_hash(atom_positions):
            return None
        return self._energy

class SystemFast:
    def __init__(
        self,
        atoms: list[Atom],
        connections: Graph,
        force_field: ForceFieldVectorized,
        temperature: float | None = None,
    ):
        with torch.no_grad():
            self.atom_positions = torch.stack([atom.position for atom in atoms]).to(torch_device)
        self.atom_positions.requires_grad = True
        self.atom_masses = torch.tensor([atom.atom_type.mass for atom in atoms], requires_grad=False).to(torch_device)
        self.atom_elements = [atom.element for atom in atoms]
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

    def initialize_velocities(self, temperature: float) -> Tensor:
        """
        Initialize velocities for the atoms in the system
        according to the Maxwell-Boltzmann distribution.
        """

        torch.manual_seed(0)

        # TODO: Verify that this is the correct way to initialize velocities
        vx = torch.normal(mean=0, std=torch.sqrt(BOLTZMANN * temperature / self.atom_masses))
        vy = torch.normal(mean=0, std=torch.sqrt(BOLTZMANN * temperature / self.atom_masses))
        vz = torch.normal(mean=0, std=torch.sqrt(BOLTZMANN * temperature / self.atom_masses))
        return torch.stack([vx, vy, vz], dim=1)


    def get_bonds_energy(self) -> Tensor:
        harmonic_bond_forces = self.force_field.harmonic_bond_forces
        if harmonic_bond_forces is None:
            return torch.tensor(0.0)

        return harmonic_bond_forces.get_forces(self.atom_positions)

    def get_angles_energy(self) -> Tensor:
        harmonic_angle_forces = self.force_field.harmonic_angle_forces
        if harmonic_angle_forces is None:
            return torch.tensor(0.0)

        return harmonic_angle_forces.get_forces(self.atom_positions)

    def get_dihedrals_energy(self) -> Tensor:
        dihedral_forces = self.force_field.dihedral_forces
        if dihedral_forces is None:
            return torch.tensor(0.0)
        return dihedral_forces.get_forces(self.atom_positions)

    def get_non_bonded_energy(self) -> Tensor:
        energy = torch.tensor(0.0).to(torch_device)
        if self.force_field.coulomb_forces is not None:
            energy += self.force_field.coulomb_forces.get_forces(self.atom_positions)
        if self.force_field.lennard_jones_forces is not None:
            energy += self.force_field.lennard_jones_forces.get_forces(self.atom_positions)
        return energy

    def get_potential_energy(self) -> Tensor:
        cached_energy = self._potential_energy_cache.get_energy(self.atom_positions)
        if cached_energy is not None:
            return cached_energy
        
        total_energy = torch.tensor(0.0).to(torch_device)
        total_energy += self.get_bonds_energy()
        total_energy += self.get_angles_energy()
        total_energy += self.get_dihedrals_energy()
        total_energy += self.get_non_bonded_energy()
        self._potential_energy_cache.update(self.atom_positions, total_energy)
        return total_energy
    
    def get_kinetic_energy(self) -> Tensor:
        velocity_norms = self.velocities.norm(dim=1)
        return 0.5 * (self.atom_masses * velocity_norms ** 2).sum()
