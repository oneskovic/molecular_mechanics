import torch
from molecular_mechanics.atom import Atom
from molecular_mechanics.forces import CoulombForce, ForceField, HarmonicAngleForce, HarmonicAngleForceParams, HarmonicBondForce, HarmonicBondForceParams, LennardJonesForce, LennardJonesForceParams


atoms = [
    Atom("H", torch.tensor([-0.180, 1.484, 1.180], requires_grad=True), 1.007947),
    Atom("O", torch.tensor([-0.648, 2.296, 0.824], requires_grad=True), 15.99943),
    Atom("H", torch.tensor([-1.592, 2.304, 1.156], requires_grad=True), 1.007947),
    Atom("H", torch.tensor([-3.268, 2.344, 2.736], requires_grad=True), 1.007947),
    Atom("O", torch.tensor([-3.264, 2.320, 1.736], requires_grad=True), 15.99943),
    Atom("H", torch.tensor([-3.736, 1.496, 1.420], requires_grad=True), 1.007947),
    Atom("H", torch.tensor([0.640, 0.076, 2.804], requires_grad=True), 1.007947),
    Atom("O", torch.tensor([0.644, 0.056, 1.804], requires_grad=True), 15.99943),
    Atom("H", torch.tensor([0.176, -0.768, 1.488], requires_grad=True), 1.007947),
]

# TIP3P-CHARMM taken from https://docs.lammps.org/Howto_tip3p.html
# LJ parameters sigma and epsilon are taken from O-O and H-H interactions
# because we use combining rules (https://en.wikipedia.org/wiki/Combining_rules)
force_field = ForceField(
    harmonic_bond_forces=HarmonicBondForce({
        ("H", "O"): HarmonicBondForceParams(0.9572, 450),
    }),
    harmonic_angle_forces=HarmonicAngleForce({
        ("H", "O", "H"): HarmonicAngleForceParams(1.82421813418, 836.8),
    }),
    lennard_jones_forces=LennardJonesForce({
        "O": LennardJonesForceParams(0.1521, 3.1507),
        "H": LennardJonesForceParams(0.0460, 0.4)
    }),
    coulomb_forces=CoulombForce({
        "O": -0.834,
        "H": 0.417
    })
)
connections = [[1], [0, 2], [1], [4], [3, 5], [4], [7], [6, 8], [7]]