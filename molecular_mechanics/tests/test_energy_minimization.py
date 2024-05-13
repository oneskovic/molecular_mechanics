import numpy as np

from molecular_mechanics.callbacks import EnergyMonitor
from molecular_mechanics.energy_minimization import minimize_energy
from molecular_mechanics.system import System


def test_convergence(three_waters):
    atoms, force_field, connections = three_waters
    system = System(atoms, connections, force_field)
    energy_monitor = EnergyMonitor()
    minimize_energy(system, callback=energy_monitor)
    last_iterations_variance = np.var(energy_monitor.potential_energies[-3:])
    assert np.isclose(last_iterations_variance, 0.0, atol=1e-2)