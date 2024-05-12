import torch
import pytest

from molecular_mechanics.atom import Atom, AtomType
from molecular_mechanics.forces import CoulombForce, DihedralForce, DihedralForceParams, ForceField, HarmonicAngleForce, HarmonicAngleForceParams, HarmonicBondForce, HarmonicBondForceParams, LennardJonesForce, LennardJonesForceParams
from molecular_mechanics.residue_database import Residue, ResidueDatabase

@pytest.fixture
def ethane():
    carbon_type = AtomType("C", "C", "C", 12.0)
    hydrogen_type = AtomType("H", "H", "H", 1.0)
    atoms = [
        Atom("C", 0, torch.tensor([0.7516, -0.0225, -0.0209], requires_grad=True), carbon_type),
        Atom("H", 0, torch.tensor([1.1851, -0.0039, 0.9875], requires_grad=True), hydrogen_type),
        Atom("H", 0, torch.tensor([1.1669, 0.8330, -0.5693], requires_grad=True), hydrogen_type),
        Atom("H", 0, torch.tensor([1.1155, -0.9329, -0.5145], requires_grad=True), hydrogen_type),
        Atom("C", 0, torch.tensor([-0.7516, 0.0225, 0.0209], requires_grad=True), carbon_type),
        Atom("H", 0, torch.tensor([-1.1669, -0.8334, 0.5687], requires_grad=True), hydrogen_type),
        Atom("H", 0, torch.tensor([-1.1157, 0.9326, 0.5151], requires_grad=True), hydrogen_type),
        Atom("H", 0, torch.tensor([-1.1850, 0.0044, -0.9875], requires_grad=True), hydrogen_type),
    ]
    force_field = ForceField(
        harmonic_bond_forces=HarmonicBondForce(
            {
                ("C", "C"): HarmonicBondForceParams(1.5375, 251),
                ("C", "H"): HarmonicBondForceParams(1.0949, 278),
            }
        ),
        harmonic_angle_forces=HarmonicAngleForce(
            {
                ("H", "C", "C"): HarmonicAngleForceParams(1.912, 388),
                ("H", "C", "H"): HarmonicAngleForceParams(1.892, 328),
            }
        ),
        dihedral_forces=DihedralForce(
            {("H", "C", "C", "H"): DihedralForceParams(0.6276, 3, 0.0)}
        )
    )
    connections = [[1, 2, 3, 4], [0], [0], [0], [0, 5, 6, 7], [4], [4], [4]]

    return atoms, force_field, connections

@pytest.fixture
def three_waters():
    hydrogen_type = AtomType("H", "H", "H", 1.007947)
    oxygen_type = AtomType("O", "O", "O", 15.99943)

    atoms = [
        Atom("H", 0.417, torch.tensor([-0.180, 1.484, 1.180], requires_grad=True), hydrogen_type),
        Atom("O", -0.834, torch.tensor([-0.648, 2.296, 0.824], requires_grad=True), oxygen_type),
        Atom("H", 0.417, torch.tensor([-1.592, 2.304, 1.156], requires_grad=True), hydrogen_type),
        Atom("H", 0.417, torch.tensor([-3.268, 2.344, 2.736], requires_grad=True), hydrogen_type),
        Atom("O", -0.834, torch.tensor([-3.264, 2.320, 1.736], requires_grad=True), oxygen_type),
        Atom("H", 0.417, torch.tensor([-3.736, 1.496, 1.420], requires_grad=True), hydrogen_type),
        Atom("H", 0.417, torch.tensor([0.640, 0.076, 2.804], requires_grad=True), hydrogen_type),
        Atom("O", -0.834, torch.tensor([0.644, 0.056, 1.804], requires_grad=True), oxygen_type),
        Atom("H", 0.417, torch.tensor([0.176, -0.768, 1.488], requires_grad=True), hydrogen_type),
    ]    

    force_field = ForceField(
        harmonic_bond_forces=HarmonicBondForce({
            ("H", "O"): HarmonicBondForceParams(0.9572, 450.0),
        }),
        harmonic_angle_forces=HarmonicAngleForce({
            ("H", "O", "H"): HarmonicAngleForceParams(1.82421813418, 55.0),
        }),
        lennard_jones_forces=LennardJonesForce({
            "O": LennardJonesForceParams(0.1521, 3.1507),
            "H": LennardJonesForceParams(0.0460, 0.4)
        }),
        coulomb_forces=CoulombForce()
    )
    connections = [[1], [0, 2], [1], [4], [3, 5], [4], [7], [6, 8], [7]]
    return atoms, force_field, connections