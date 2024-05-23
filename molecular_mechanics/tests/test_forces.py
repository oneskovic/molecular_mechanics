import pytest
import torch
from torch.testing import assert_close

from molecular_mechanics.constants import COULOMB
from molecular_mechanics.forces import HarmonicBondForceParams, SoftSearch

def test_bond_force(ethane):
    atoms, forcefield, _ = ethane
    bond_forces = forcefield.harmonic_bond_forces

    force_ch = bond_forces.get_force(atoms[0], atoms[1])
    assert_close(force_ch, torch.tensor(0.001159))
    
    force_cc = bond_forces.get_force(atoms[0], atoms[4])
    assert_close(force_cc, torch.tensor(0.137049))

def test_bond_force_symmetry(ethane):
    atoms, forcefield, _ = ethane
    bond_forces = forcefield.harmonic_bond_forces

    force_12 = bond_forces.get_force(atoms[0], atoms[1])
    force_21 = bond_forces.get_force(atoms[1], atoms[0])
    assert_close(force_12, force_21)

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

def test_angle_force_symmetry(ethane):
    atoms, forcefield, _ = ethane
    angle_forces = forcefield.harmonic_angle_forces

    force_hch = angle_forces.get_force(atoms[1], atoms[0], atoms[2])
    force_hhc = angle_forces.get_force(atoms[2], atoms[0], atoms[1])
    assert_close(force_hch, force_hhc)

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

def test_dihedral_force_symmetry(ethane):
    atoms, forcefield, _ = ethane
    dihedral_forces = forcefield.dihedral_forces

    force_hcch = dihedral_forces.get_force(atoms[1], atoms[0], atoms[4], atoms[5])
    force_hcch_sym = dihedral_forces.get_force(atoms[5], atoms[4], atoms[0], atoms[1])
    assert_close(force_hcch, force_hcch_sym)

def test_dihedral_force_not_found(ethane):
    atoms, forcefield, _ = ethane
    dihedral_forces = forcefield.dihedral_forces
    force = dihedral_forces.get_force(atoms[0], atoms[1], atoms[2], atoms[3])
    assert_close(force, torch.tensor(0.0))

def test_lennard_jones_force(three_waters):
    atoms, forcefield, _ = three_waters
    lj_forces = forcefield.lennard_jones_forces

    force_oh = lj_forces.get_force(atoms[0], atoms[4])
    assert_close(force_oh, torch.tensor(-0.008758))
    force_hh = lj_forces.get_force(atoms[0], atoms[3])
    # TODO: The force is very small between H-H, should
    # test this with different positions
    assert_close(force_hh, torch.tensor(0.0000003))

    force_oo = lj_forces.get_force(atoms[1], atoms[4])
    assert_close(force_oo, torch.tensor(1.530663))

def test_lennard_jones_force_symmetry(three_waters):
    atoms, forcefield, _ = three_waters
    lj_forces = forcefield.lennard_jones_forces

    force_oh = lj_forces.get_force(atoms[0], atoms[4])
    force_ho = lj_forces.get_force(atoms[4], atoms[0])
    assert_close(force_oh, force_ho)

def test_coulomb_force(three_waters):
    atoms, forcefield, _ = three_waters
    coulomb_forces = forcefield.coulomb_forces

    force_oh = coulomb_forces.get_force(atoms[2], atoms[4])
    assert_close(force_oh, torch.tensor(-0.1965054455*COULOMB))

def test_coulomb_force_symmetry(three_waters):
    atoms, forcefield, _ = three_waters
    coulomb_forces = forcefield.coulomb_forces

    force_oh = coulomb_forces.get_force(atoms[2], atoms[4])
    force_ho = coulomb_forces.get_force(atoms[4], atoms[2])
    assert_close(force_oh, force_ho)

def test_soft_searcher():
    atoms = ("C", "H")
    bond_dict = {
        ("C1", "HC1"): HarmonicBondForceParams(0.0, 0.0),
        ("N", "H"): HarmonicBondForceParams(0.0, 0.0),
    }
    soft_searcher = SoftSearch(bond_dict)
    key = soft_searcher.search(atoms)
    assert key == ("C1", "HC1")