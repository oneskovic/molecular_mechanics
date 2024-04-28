from molecular_mechanics.atom import get_bond_angle, get_dihedral_angle, get_distance
from molecular_mechanics.system import System

def print_system_state(system: System) -> None:
    description_width = 25
    value_width = 20
    precision = 6

    header = "STATE"
    width = description_width + value_width
    dashes = "-" * ((width - len(header)) // 2)
    print(dashes + header + dashes)

    desc_val_dict: dict[str, float] = {}

    potential_energy = system.get_potential_energy()
    desc_val_dict["Total energy"] = potential_energy.item()

    # FIXME: Duplicate code from get_bonds_energy
    for bond in system.bonds:
        ind1, ind2 = bond
        atom1, atom2 = system.atoms[ind1], system.atoms[ind2]
        description = f"Bond length {atom1.element}{ind1}-{atom2.element}{ind2}:"
        distance = get_distance(atom1, atom2)
        desc_val_dict[description] = distance.item()

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
        print(
            f"{description:<{description_width}}{value:>{value_width}.{precision}f}"
        )
    print("-" * width)


class XYZTrajectoryWriter:
    def __init__(self, filename: str, system: System) -> None:
        self._system = system
        self._file = open(filename, "w")
        

    def write(self) -> None:
        self._file.write(f"{len(self._system.atoms)}\n")  # number of atoms
        self._file.write("\n")  # name of molecule
        for atom in self._system.atoms:
            self._file.write(f"{atom.element} {atom.position[0].item():6f} {atom.position[1].item():6f} {atom.position[2].item():6f}\n")


    def close(self) -> None:
        self._file.close()