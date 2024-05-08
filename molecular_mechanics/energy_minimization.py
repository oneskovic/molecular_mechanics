from molecular_mechanics.callbacks import Callback
from molecular_mechanics.system import System
from torch.optim import Adam


def minimize_energy(system: System, max_iterations: int = 1000, callback: Callback | None = None):
    positions = [atom.position for atom in system.atoms]
    optimizer = Adam(positions)

    for i in range(max_iterations):
        energy = system.get_potential_energy()
        if energy < 1e-4:
            break
        energy.backward()
        optimizer.step()
        optimizer.zero_grad()
        if callback is not None:
            callback(i, system)
    
    if callback is not None:
        callback.close()
