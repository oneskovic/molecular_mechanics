import itertools

import torch
from torch import Tensor

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
        self.temperature = temperature

        # Precomputed properties of the system
        self.bonds = get_all_bonds(connections)
        self.angles = get_all_angles(connections)
        self.dihedrals = get_all_dihedrals(connections)
        self.all_pairs_bond_separation = get_all_pairs_bond_separation(connections)

        if self.temperature is not None:
            self.velocities = self.initialize_velocities(temperature)

        self._potential_energy_cache = _PotentialEnergyCache()

    def initialize_velocities(self, temperature):
        """
        Initialize velocities for the atoms in the system
        according to the Maxwell-Boltzmann distribution.
        """

        torch.manual_seed(0)

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

    def get_non_bonded_energy(self) -> Tensor:
        lennard_jones_forces = self.force_field.lennard_jones_forces
        coulomb_forces = self.force_field.coulomb_forces
        scaling_factor = self.force_field.non_bonded_scaling_factor
        total_energy = torch.tensor(0.0)
        for (i, atom1), (j, atom2) in itertools.combinations(enumerate(self.atoms), 2):
            bond_separation = self.all_pairs_bond_separation[i][j]
            lj = (
                lennard_jones_forces.get_force(atom1, atom2)
                if lennard_jones_forces
                else torch.tensor(0)
            )
            coulomb = (
                coulomb_forces.get_force(atom1, atom2)
                if coulomb_forces
                else torch.tensor(0)
            )
            # Nonbonded interactions are only calculated between atoms in
            # different molecules or for atoms in the same molecule separated
            # by at least three bonds. Those non-bonded interactions separated
            # by exactly three bonds (“1-4 interactions”) are reduced by the
            # application of a scalefactor.
            if bond_separation < 3:
                continue
            if scaling_factor and bond_separation == 3:
                total_energy += lj * scaling_factor
                total_energy += coulomb * scaling_factor
            else:
                total_energy += lj
                total_energy += coulomb
        return total_energy

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
