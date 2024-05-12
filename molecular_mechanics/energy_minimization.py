from molecular_mechanics.callbacks import Callback
from molecular_mechanics.system import System
from torch.optim import Adam


def minimize_energy(system: System, max_iterations: int = 1000, callback: Callback | None = None):
    positions = [atom.position for atom in system.atoms]
    optimizer = Adam(positions)

    energy = system.get_potential_energy()
    for i in range(max_iterations):
        energy.backward()
        optimizer.step()
        optimizer.zero_grad()
        new_energy = system.get_potential_energy()
        if abs(energy - new_energy) < 1e-3:
            break
        energy = new_energy
        if callback is not None:
            callback(i, system)
    
    if callback is not None:
        callback.close()
