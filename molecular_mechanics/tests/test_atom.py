import torch

from molecular_mechanics.atom import get_bond_angle, get_dihedral_angle, get_distance

def test_bond_angle(ethane):
    atoms, _, _ = ethane
    # H-C-H
    torch.testing.assert_close(get_bond_angle(atoms[1], atoms[0], atoms[2]), torch.tensor(1.8715999))
    # H-C-C
    torch.testing.assert_close(get_bond_angle(atoms[1], atoms[0], atoms[4]), torch.tensor(1.9482204))

def test_get_distance(ethane):
    atoms, _, _ = ethane
    # H-C
    torch.testing.assert_close(get_distance(atoms[0], atoms[1]), torch.tensor(1.097788))
    # C-C
    torch.testing.assert_close(get_distance(atoms[0], atoms[4]), torch.tensor(1.504454))

def test_dihedral_angle(ethane):
    atoms, _, _ = ethane
    # H-C-C-H
    torch.testing.assert_close(get_dihedral_angle(atoms[1], atoms[0], atoms[4], atoms[5]), torch.tensor(1.0478654))