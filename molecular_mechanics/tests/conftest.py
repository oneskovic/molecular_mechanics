import torch
import pytest

from molecular_mechanics.atom import Atom
from molecular_mechanics.forces import DihedralForce, DihedralForceParams, ForceField, HarmonicAngleForce, HarmonicAngleForceParams, HarmonicBondForce, HarmonicBondForceParams

@pytest.fixture
def ethane():
    atoms = [
        Atom("C", torch.tensor(
            [0.7516, -0.0225, -0.0209], requires_grad=True), 12.0),
        Atom("H", torch.tensor(
            [1.1851, -0.0039, 0.9875], requires_grad=True), 1.0),
        Atom("H", torch.tensor(
            [1.1669, 0.8330, -0.5693], requires_grad=True), 1.0),
        Atom("H", torch.tensor(
            [1.1155, -0.9329, -0.5145], requires_grad=True), 1.0),
        Atom("C", torch.tensor(
            [-0.7516, 0.0225, 0.0209], requires_grad=True), 12.0),
        Atom("H", torch.tensor(
            [-1.1669, -0.8334, 0.5687], requires_grad=True), 1.0),
        Atom("H", torch.tensor(
            [-1.1157, 0.9326, 0.5151], requires_grad=True), 1.0),
        Atom("H", torch.tensor(
            [-1.1850, 0.0044, -0.9875], requires_grad=True), 1.0),
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
    connections = [[0], [1, 2, 3, 4], [0], [0], [0, 5, 6, 7], [4], [4], [4]]
    return atoms, force_field, connections