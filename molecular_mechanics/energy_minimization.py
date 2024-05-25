import sys

from molecular_mechanics.callbacks import Callback
from molecular_mechanics.system import System
from molecular_mechanics.system_fast import SystemFast
from torch.optim import LBFGS

def minimize_energy(system: System | SystemFast, max_iterations: int | None = None, callback: Callback | None = None):
    if max_iterations is None:
        max_iterations = sys.maxsize
    
    if isinstance(system, System):
        positions = [atom.position for atom in system.atoms]
    elif isinstance(system, SystemFast):
        positions = [system.atom_positions]
    optimizer = LBFGS(positions,history_size=10, max_iter=4)
    def closure():
        optimizer.zero_grad()
        if isinstance(system, System):
            energy = system.get_potential_energy()
        else:
            energy = system.get_potential_energy(use_cache=False)
        energy.backward(retain_graph=True)
        return energy.item()


    if callback is not None:
        callback(0, system)

    energy = system.get_potential_energy()
    for i in range(1, max_iterations + 1):
        optimizer.step(closure)
        if callback is not None:
            callback(i, system)
        new_energy = system.get_potential_energy()
        if abs(energy - new_energy) < 0.1:
            break
        energy = new_energy
    
    if callback is not None:
        callback.close()