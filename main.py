from atom import Atom
from forces import (
    ForceField,
    HarmonicBondForce,
    HarmonicBondForceParams,
    HarmonicAngleForce,
    HarmonicAngleForceParams,
    LennardJonesForce,
    LennardJonesForceParams,
    CoulombForce,
)
from system import System

import torch
from torch.optim import Adam


atoms = [
    Atom("H", torch.tensor([0.0, 1, 0.0], requires_grad=True), 1.0),
    Atom("O", torch.tensor([0.0, 0.0, 0.0], requires_grad=True), 16.0),
    Atom("H", torch.tensor([0.0, 0.0, 1], requires_grad=True), 1.0),
]
force_field = ForceField(
    harmonic_bond_forces=HarmonicBondForce(
        {("O", "H"): HarmonicBondForceParams(0.09572, 462750.4)}
    ),
    harmonic_angle_forces=HarmonicAngleForce(
        {("H", "O", "H"): HarmonicAngleForceParams(1.82421813418, 836.8)}
    ),
    lennard_jones_forces=LennardJonesForce(
        {
            "H": LennardJonesForceParams(0.0, 1.0),
            "O": LennardJonesForceParams(0.635968, 0.31507524065751241),
        }
    ),
    coulomb_forces=CoulombForce({"H": 0.417, "O": -0.834}),
)
connections = [[1], [0, 2], [1]]

iterations = 5000
system = System(atoms, connections, force_field)
positions = [atom.position for atom in system.atoms]
optimizer = Adam(positions)

for i in range(iterations):
    if i % 100 == 0:
        print(f"Iteration {i}")
        system.print_state()
    energy = system.get_total_energy()
    energy.backward()
    optimizer.step()
    optimizer.zero_grad()
