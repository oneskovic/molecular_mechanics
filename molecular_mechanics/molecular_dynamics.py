from molecular_mechanics.callbacks import Callback
from molecular_mechanics.integration import VerletIntegrator
from molecular_mechanics.system import System


def run_dynamics(system: System, iterations: int, callback: Callback | None = None):
    integrator = VerletIntegrator(system)
    for i in range(iterations):
        integrator.step()
        if callback is not None:
            callback(i, system)

    if callback is not None:
        callback.close()