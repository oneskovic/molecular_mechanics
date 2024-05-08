import pytest
import torch
from torch.testing import assert_close

def test_bond_force(ethane):
    atoms, forcefield, _ = ethane
    bond_forces = forcefield.harmonic_bond_forces

    force_ch = bond_forces.get_force(atoms[0], atoms[1])
    assert_close(force_ch, torch.tensor(0.001159))
    
    force_cc = bond_forces.get_force(atoms[0], atoms[4])
    assert_close(force_cc, torch.tensor(0.137049))

def test_bond_force_not_found(ethane):
    atoms, forcefield, _ = ethane
    bond_forces = forcefield.harmonic_bond_forces

    with pytest.raises(KeyError):
        bond_forces.get_force(atoms[1], atoms[2])

def test_angle_force(ethane):
    atoms, forcefield, _ = ethane
    angle_forces = forcefield.harmonic_angle_forces

    force_hch = angle_forces.get_force(atoms[1], atoms[0], atoms[2])
    assert_close(force_hch, torch.tensor(0.068251))

    force_hcc = angle_forces.get_force(atoms[1], atoms[0], atoms[4])
    assert_close(force_hcc, torch.tensor(0.254513))

def test_angle_force_not_found(ethane):
    atoms, forcefield, _ = ethane
    angle_forces = forcefield.harmonic_angle_forces
    with pytest.raises(KeyError):
        angle_forces.get_force(atoms[1], atoms[2], atoms[3])

def test_dihedral_force(ethane):
    atoms, forcefield, _ = ethane
    dihedral_forces = forcefield.dihedral_forces

    force_hcch = dihedral_forces.get_force(atoms[1], atoms[0], atoms[4], atoms[5])
    assert_close(force_hcch, torch.tensor(0.000001))

def test_dihedral_force_not_found(ethane):
    atoms, forcefield, _ = ethane
    dihedral_forces = forcefield.dihedral_forces
    with pytest.raises(KeyError):
        dihedral_forces.get_force(atoms[0], atoms[1], atoms[2], atoms[3])

# TODO: Add tests for non-bonded forces