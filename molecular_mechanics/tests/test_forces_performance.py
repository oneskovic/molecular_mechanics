import numpy as np

from molecular_mechanics.atom import Atom
from molecular_mechanics.callbacks import EnergyMonitor
from molecular_mechanics.forces import ForceField
from molecular_mechanics.molecular_dynamics import run_dynamics
from molecular_mechanics.molecule import Graph
from molecular_mechanics.system import System
from molecular_mechanics.forces_fast import HarmonicBondForce
import torch

def harmonic_bond_force_old(molecule: tuple[list[Atom], ForceField, Graph]):
        atoms, forcefield, connections = molecule
        system = System(atoms, connections, forcefield)

        for _ in range(100):
            system.get_bonds_energy().backward()
            with torch.no_grad():
                for atom in atoms:
                    atom.position -= 0.1 * atom.position.grad
            atom.position.grad.zero_()

def harmonic_bond_force_new(molecule: tuple[list[Atom], ForceField, Graph]):
        atoms, forcefield, connections = molecule
        system = System(atoms, connections, forcefield)
        fast_bond_force = HarmonicBondForce(forcefield.harmonic_bond_forces.bond_dict, system.bonds, atoms)

        for _ in range(100):
            fast_bond_force.get_forces(atoms).backward()
            with torch.no_grad():
                for atom in atoms:
                    atom.position -= 0.1 * atom.position.grad
            atom.position.grad.zero_()
                
def test_harmonic_bond_water_old(benchmark, three_waters):
    benchmark(harmonic_bond_force_old, three_waters)

def test_harmonic_bond_water_new(benchmark, three_waters):
    benchmark(harmonic_bond_force_new, three_waters)
