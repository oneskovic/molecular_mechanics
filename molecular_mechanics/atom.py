import torch
from torch import Tensor

class AtomType:
    def __init__(self, element : str, name : str, atom_class : str, mass : float):
        self.element = element
        self.name = name
        self.atom_class = atom_class
        self.mass = mass

class Atom:
    def __init__(self, element: str, charge: float, position: Tensor, atom_type: AtomType):
        self.element = element
        self.position = position
        self.atom_type = atom_type
        self.charge = charge


def get_bond_angle(atom1: Atom, atom2: Atom, atom3: Atom) -> Tensor:
    vec12 = atom1.position - atom2.position
    vec32 = atom3.position - atom2.position
    return torch.acos(
        vec12.dot(vec32) / (get_distance(atom1, atom2) * get_distance(atom2, atom3))
    )

def get_distance(atom1: Atom, atom2: Atom) -> Tensor:
    return (atom1.position - atom2.position).norm()

def get_dihedral_angle(atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom) -> Tensor:
    # https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    b1 = atom2.position - atom1.position
    b2 = atom3.position - atom2.position
    b3 = atom4.position - atom3.position

    cross12 = torch.linalg.cross(b1, b2)
    cross23 = torch.linalg.cross(b2, b3)

    n1 = cross12 / torch.linalg.norm(cross12)
    n2 = cross23 / torch.linalg.norm(cross23)

    b2_normalized = b2 / torch.linalg.norm(b2)
    m1 = torch.linalg.cross(n1, b2_normalized)

    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)

    angle = torch.atan2(y, x)
    return angle

def get_dihedral_angle_positions(atom1_position: torch.Tensor, atom2_position: torch.Tensor, atom3_position: torch.Tensor, atom4_position: torch.Tensor) -> Tensor:
    # https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    b1 = atom2_position - atom1_position
    b2 = atom3_position - atom2_position
    b3 = atom4_position - atom3_position

    cross12 = torch.cross(b1, b2)
    cross23 = torch.cross(b2, b3)

    n1 = cross12 / torch.linalg.norm(cross12)
    n2 = cross23 / torch.linalg.norm(cross23)

    b2_normalized = b2 / torch.linalg.norm(b2)
    m1 = torch.cross(n1, b2_normalized)

    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)

    angle = torch.atan2(y, x)
    return angle
