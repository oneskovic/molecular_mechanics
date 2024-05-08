import torch
from molecular_mechanics.atom import Atom
from molecular_mechanics.forces import ForceField, HarmonicAngleForce, HarmonicAngleForceParams, HarmonicBondForce, HarmonicBondForceParams


atoms = [
    Atom("H", torch.tensor(
        [-22.157, 18.401, -21.626], requires_grad=True), 1.0),
    Atom("O", torch.tensor(
        [-23.107, 18.401, -21.626], requires_grad=True), 16.0),
    Atom("H", torch.tensor(
        [-23.424, 18.401, -20.730], requires_grad=True), 1.0),
]
force_field = ForceField(
    harmonic_bond_forces=HarmonicBondForce(
        {("O", "H"): HarmonicBondForceParams(0.9572, 553.0)}
    ),
    harmonic_angle_forces=HarmonicAngleForce(
        {("H", "O", "H"): HarmonicAngleForceParams(1.82421813418, 100)}
    )
)
connections = [[1], [0, 2], [1]]