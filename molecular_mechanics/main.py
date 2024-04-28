import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
from molecular_mechanics.forcefield_parser import ff14sb_forcefield
from molecular_mechanics.atom import Atom, get_bond_angle, get_distance
from molecular_mechanics.integration import VerletIntegrator
from molecular_mechanics.system import System
from molecular_mechanics.pdb_parser import atoms_and_bonds_from_pdb

force_field = ff14sb_forcefield()
atoms, connections = atoms_and_bonds_from_pdb("data/vodica.pdb", force_field)
system = System(atoms, connections, force_field, temperature=0.0)

def minimize_energy(system: System, iterations: int):
    positions = [atom.position for atom in system.atoms]
    optimizer = Adam(positions)

    for i in range(iterations):
        if i % 100 == 0:
            print(f"Iteration {i}")
            system.print_state()
        energy = system.get_potential_energy()
        energy.backward()
        optimizer.step()
        # Invalidate the cached total energy
        system.potential_energy = None
        optimizer.zero_grad()

def dynamics(system: System, iterations: int):
    integrator = VerletIntegrator(system, timestep=0.00001)
    print_freq = 100
    total_energy = []
    potential_energy = []
    kinetic_energy = []
    bond_angle = []
    bond_length12 = []
    bond_length23 = []
    for i in range(iterations):
        p = system.get_potential_energy()
        k = system.get_kinetic_energy()
        total_energy.append((p + k).item())
        potential_energy.append(p.item())
        kinetic_energy.append(k.item())
        #bond_angle.append(get_bond_angle(*system.atoms).item())
        bond_length12.append(get_distance(system.atoms[0], system.atoms[1]).item())
        bond_length23.append(get_distance(system.atoms[1], system.atoms[2]).item())
        if i % print_freq == 0:
            print(f"Iteration {i}")
            system.print_state()
        integrator.step()
    

    fig, axs = plt.subplot_mosaic([
        ["energy", "energy"],
        ["bond_angle", "bond_lengths"]
    ])

    axs["energy"].set_title("Energies")
    axs["energy"].plot(total_energy, label="Total Energy")
    axs["energy"].plot(potential_energy, label="Potential Energy")
    axs["energy"].plot(kinetic_energy, label="Kinetic Energy")
    axs["energy"].legend()

    axs["bond_angle"].set_title("Bond Angle")
    axs["bond_angle"].plot(bond_angle, label="Bond Angle")

    axs["bond_lengths"].set_title("Bond Lengths")
    axs["bond_lengths"].plot(bond_length12, label="Bond Length 1-2")
    axs["bond_lengths"].plot(bond_length23, label="Bond Length 2-3")
    axs["bond_lengths"].legend()
    plt.show()

if __name__ == "__main__":
    dynamics(system, 10000)