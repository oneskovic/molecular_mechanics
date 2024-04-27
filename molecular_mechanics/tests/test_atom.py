import torch
from pytest import approx

from molecular_mechanics.atom import Atom, get_bond_angle


def test_bond_angle():
    atom1 = Atom("H", torch.tensor([0.0, -1.0, 0.0]), 1.0)
    atom2 = Atom("O", torch.tensor([2.0, 0.0, 0.0]), 16.0)
    atom3 = Atom("H", torch.tensor([0.0, 0.0, 2.0]), 1.0)
    assert get_bond_angle(atom1, atom2, atom3) == approx(0.8860771238)