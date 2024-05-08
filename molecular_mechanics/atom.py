import torch
from torch import Tensor


class Atom:
    def __init__(self, element: str, position: Tensor, mass: float):
        self.element = element
        self.position = position
        self.mass = mass


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
