import sys

from molecular_mechanics.callbacks import Callback
from molecular_mechanics.system import System
from torch.optim import LBFGS

def minimize_energy(system: System, max_iterations: int | None = None, callback: Callback | None = None):
    if max_iterations is None:
        max_iterations = sys.maxsize
    positions = [atom.position for atom in system.atoms]
    optimizer = LBFGS(positions)
    def closure():
        optimizer.zero_grad()
        energy = system.get_potential_energy()
        energy.backward()
        return energy.item()


    if callback is not None:
        callback(0, system)

    energy = system.get_potential_energy()
    for i in range(1, max_iterations + 1):
        optimizer.step(closure)
        if callback is not None:
            callback(i, system)
        new_energy = system.get_potential_energy()
        if abs(energy - new_energy) < 1e-3:
            break
        energy = new_energy
    
    if callback is not None:
        callback.close()