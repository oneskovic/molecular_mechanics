from atom import Atom, get_bond_angle, get_distance
from forces import ForceField
from molecule import get_all_bonds, get_all_pairs_bond_separation
from torch import Tensor
import torch


class System:
    def __init__(
        self,
        atoms: list[Atom],
        connections: list[list[int]],
        force_field: ForceField,
    ):
        self.atoms = atoms
        self.connections = connections
        self.force_field = force_field
        self.bonds = get_all_bonds(connections)
        self.all_pairs_bond_separation = get_all_pairs_bond_separation(connections)
        self.total_energy = None

    def get_bonds_energy(self) -> Tensor:
        harmonic_bond_forces = self.force_field.harmonic_bond_forces
        if harmonic_bond_forces is None:
            return torch.tensor(0.0)

        energy = torch.tensor(0.0)
        for bond in self.bonds:
            ind1, ind2 = bond
            if ind1 < ind2:
                atom1, atom2 = self.atoms[ind1], self.atoms[ind2]
                energy += harmonic_bond_forces.get_force(atom1, atom2)
        return energy

    def get_angles_energy(self) -> Tensor:
        harmonic_angle_forces = self.force_field.harmonic_angle_forces
        if harmonic_angle_forces is None:
            return torch.tensor(0.0)

        energy = torch.tensor(0.0)
        # FIXME: Refactor this to iterate over all angles
        # similarly to how bonds are handled

        # Iterate over every "center" atom
        for j, atom2 in enumerate(self.atoms):
            # Iterate over all pairs of neighbors
            for i in self.connections[j]:
                atom1 = self.atoms[i]
                for k in self.connections[j]:
                    atom3 = self.atoms[k]
                    # Ensure bonds are not double counted
                    if i < k:
                        energy += harmonic_angle_forces.get_force(atom1, atom2, atom3)
        return energy

    def get_non_bonded_energy(self) -> Tensor:
        lennard_jones_forces = self.force_field.lennard_jones_forces
        coulomb_forces = self.force_field.coulomb_forces
        scaling_factor = self.force_field.non_bonded_scaling_factor
        total_energy = torch.tensor(0.0)
        for i, atom1 in enumerate(self.atoms):
            for j, atom2 in enumerate(self.atoms):
                # Ensure bonds are not double counted
                if i >= j:
                    continue
                bond_separation = self.all_pairs_bond_separation[i][j]
                lj = (
                    lennard_jones_forces.get_force(atom1, atom2)
                    if lennard_jones_forces
                    else 0
                )
                coulomb = (
                    coulomb_forces.get_force(atom1, atom2) if coulomb_forces else 0
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

    def get_total_energy(self) -> Tensor:
        total_energy = torch.tensor(0.0)
        total_energy += self.get_bonds_energy()
        total_energy += self.get_angles_energy()
        total_energy += self.get_non_bonded_energy()
        self.total_energy = total_energy
        return total_energy

    def print_state(self):
        description_width = 25
        value_width = 20
        precision = 6

        header = "STATE"
        width = description_width + value_width
        dashes = "-" * ((width - len(header)) // 2)
        print(dashes + header + dashes)

        desc_val_dict: dict[str, float] = {}

        if self.total_energy is None:
            self.total_energy = self.get_total_energy()
        desc_val_dict["Total energy"] = self.total_energy.item()

        # FIXME: Duplicate code from get_bonds_energy
        for bond in self.bonds:
            ind1, ind2 = bond
            if ind1 < ind2:
                atom1, atom2 = self.atoms[ind1], self.atoms[ind2]
                description = (
                    f"Bond length {atom1.element}{ind1}-{atom2.element}{ind2}:"
                )
                distance = get_distance(atom1, atom2)
                desc_val_dict[description] = distance.item()

        # FIXME: Duplicate code from get_angles_energy
        for j, atom2 in enumerate(self.atoms):
            for i in self.connections[j]:
                atom1 = self.atoms[i]
                for k in self.connections[j]:
                    atom3 = self.atoms[k]
                    if i < k:
                        atom1_str = f"{atom1.element}{i}"
                        atom2_str = f"{atom2.element}{j}"
                        atom3_str = f"{atom3.element}{k}"
                        description = f"Angle {atom1_str}-{atom2_str}-{atom3_str}:"
                        angle = get_bond_angle(atom1, atom2, atom3)
                        desc_val_dict[description] = angle.item()

        for description, value in desc_val_dict.items():
            print(
                f"{description:<{description_width}}{value:>{value_width}.{precision}f}"
            )
        print("-" * width)
