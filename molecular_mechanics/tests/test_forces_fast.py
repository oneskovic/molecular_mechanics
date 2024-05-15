import numpy as np

from molecular_mechanics.atom import Atom
from molecular_mechanics.callbacks import EnergyMonitor
from molecular_mechanics.forces import ForceField
from molecular_mechanics.molecular_dynamics import run_dynamics
from molecular_mechanics.molecule import Graph
from molecular_mechanics.system import System
from molecular_mechanics.forces_fast import HarmonicBondForce
import torch

def total_energies_on_molecule(molecule: tuple[list[Atom], ForceField, Graph]):
        atoms, forcefield, connections = molecule
        system = System(atoms, connections, forcefield)
        for _ in range(100):
            with torch.no_grad():
                bonds_energy_slow = system.get_bonds_energy()
                fast_bond_force = HarmonicBondForce(forcefield.harmonic_bond_forces.bond_dict, system.bonds, atoms)
                bonds_energy_fast = fast_bond_force.get_forces(atoms)
                abs_diff = np.abs(bonds_energy_slow - bonds_energy_fast)
                percentage_diff = abs_diff / bonds_energy_slow
                assert percentage_diff < 0.1
            run_dynamics(system, 1)

def test_three_waters(three_waters):
    total_energies_on_molecule(three_waters)

def test_ethane(ethane):
    total_energies_on_molecule(ethane)