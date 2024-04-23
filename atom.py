from torch import Tensor
import torch


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
