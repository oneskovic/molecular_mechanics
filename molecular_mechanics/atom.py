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
    vec12 = atom1.position - atom2.position
    vec32 = atom3.position - atom2.position
    vec43 = atom4.position - atom3.position

    normal1 = vec12.cross(vec32)
    normal2 = vec32.cross(vec43)

    angle = torch.acos(normal1.dot(normal2) / (normal1.norm() * normal2.norm()))
    return angle
