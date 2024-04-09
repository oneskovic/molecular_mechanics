from atom import Atom
import torch
from torch.optim import Adam
from forces import HarmonicBondForce, HarmonicBondForceParams, HarmonicAngleForce, HarmonicAngleForceParams, LennardJonesForce, LennardJonesForceParams, CoulombForce
atoms = [
    Atom("H", torch.tensor([0.0, -1.0, 0.0], requires_grad=True), 1.0),
    Atom("O", torch.tensor([2.0, 0.0, 0.0], requires_grad=True), 16.0), 
    Atom("H", torch.tensor([0.0, 0.0, 2.0], requires_grad=True), 1.0)]

harmonic_bond_force = HarmonicBondForce({
    ("O", "H"): HarmonicBondForceParams(0.09572, 462750.4)
})
harmonic_angle_force = HarmonicAngleForce({
    ("H", "O", "H"): HarmonicAngleForceParams(1.82421813418, 836.8)
})
lennard_jones_force = LennardJonesForce({
    "H": LennardJonesForceParams(0.0, 1.0),
    "O": LennardJonesForceParams(0.635968, 0.31507524065751241)
})
coulomb_force = CoulombForce({
    "H": 0.417,
    "O": -0.834
})
bonds = [(0, 1), (1, 0), (1, 2), (2, 1)]
adj = [[1],[0,2],[1]]

iterations = 1000
positions = [atom.position for atom in atoms]
optimizer = Adam(positions)
for _ in range(iterations):
    total_energy = torch.tensor(0.0)
    # Calculate energy for bonds
    for bond in bonds:
        ind1, ind2 = bond
        if ind1 < ind2:
            atom1, atom2 = atoms[ind1], atoms[ind2]
            total_energy += harmonic_bond_force.get_force(atom1, atom2)
    
    # Calculate non-bonded forces
    for i, atom1 in enumerate(atoms):
        for j, atom2 in enumerate(atoms):
            if i < j:
                total_energy += lennard_jones_force.get_force(atom1, atom2)
                total_energy += coulomb_force.get_force(atom1, atom2)
    
    # Calculate angle forces
    # Iterate over every "center" atom
    for i, atom1 in enumerate(atoms):
        # Iterate over all pairs of neighbors
        for j in adj[i]:
            atom2 = atoms[j]
            for k in adj[i]:
                atom3 = atoms[k]
                # Ensure bonds are not double counted
                if j < k:
                    total_energy += harmonic_angle_force.get_force(atom2, atom1, atom3)
    
    print(total_energy.item())
    total_energy.backward()
    # Update positions
    optimizer.step()
    optimizer.zero_grad()
