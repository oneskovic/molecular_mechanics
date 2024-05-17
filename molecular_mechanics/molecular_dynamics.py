from molecular_mechanics.callbacks import Callback
from molecular_mechanics.integration import VerletIntegrator, FastVerletIntegrator
from molecular_mechanics.system import System
from molecular_mechanics.system_fast import SystemFast


def run_dynamics(system: System | SystemFast, iterations: int, callback: Callback | None = None):
    if isinstance(system, System):
        integrator = VerletIntegrator(system)
    elif isinstance(system, SystemFast):
        integrator = FastVerletIntegrator(system)
    else:
        raise ValueError("Unknown system type")
    
    for i in range(iterations):
        integrator.step()
        if callback is not None:
            callback(i, system)

    if callback is not None:
        callback.close()
