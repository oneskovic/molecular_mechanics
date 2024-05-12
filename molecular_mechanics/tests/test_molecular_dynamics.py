import numpy as np

from molecular_mechanics.atom import Atom
from molecular_mechanics.callbacks import EnergyMonitor
from molecular_mechanics.forces import ForceField
from molecular_mechanics.molecular_dynamics import run_dynamics
from molecular_mechanics.molecule import Graph
from molecular_mechanics.system import System


def total_energies_on_molecule(molecule: tuple[list[Atom], ForceField, Graph]):
    atoms, forcefield, connections = molecule
    system = System(atoms, connections, forcefield)
    energy_monitor = EnergyMonitor()
    run_dynamics(system, iterations=1000, callback=energy_monitor)
    return np.array(energy_monitor.potential_energies) + np.array(energy_monitor.kinetic_energies)

def test_three_waters(three_waters):
    assert np.isclose(np.var(total_energies_on_molecule(three_waters)), 0.0, atol=1e-4)

def test_ethane(ethane):
    assert np.isclose(np.var(total_energies_on_molecule(ethane)), 0.0, atol=1e-4)