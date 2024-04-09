from torch import Tensor
import torch
class Atom:
    def __init__(self, element: str, position: Tensor, mass: float):
        self.element = element
        self.position = position
        self.mass = mass

def get_bond_angle(atom1: Atom, atom2: Atom, atom3: Atom) -> float:
    return torch.acos(
        (atom1.position - atom2.position).dot(atom3.position - atom2.position) /
        ((atom1.position - atom2.position).norm() * (atom3.position - atom2.position).norm())
    )