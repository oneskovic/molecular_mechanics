from typing import TypeGuard

import torch
from torch import Tensor

from molecular_mechanics.system import System

class VerletIntegrator:
    """
    Verlet integrator for molecular dynamics.

    Using the Velocity Verlet variant without half-steps.
    Described in the Wikipedia article:
    https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
    """

    def __init__(self, system: System, timestep: float = 0.001):
        self.system = system
        self.timestep = timestep

        self.mass = torch.tensor([atom.atom_type.mass for atom in self.system.atoms])

    def _get_acceleration(self) -> Tensor:
        def is_valid_grad(grad_list: list[Tensor | None]) -> TypeGuard[list[Tensor]]:
            return all(grad is not None for grad in grad_list)
        potential_energy = self.system.get_potential_energy()
        grads_list = [atom.position.grad for atom in self.system.atoms]
        if not is_valid_grad(grads_list):
            potential_energy.backward()
            grads_list = [atom.position.grad for atom in self.system.atoms]
            assert is_valid_grad(grads_list)
            
        grad = torch.stack(grads_list)
        force = -grad
        acceleration = force / self.mass[:, None]
        return acceleration
    
    def step(self) -> None:
        """
        Perform a single iteration of the Verlet integration algorithm.
        """
        acceleration = self._get_acceleration()
        position_delta = self.timestep * self.system.velocities + 0.5 * acceleration * self.timestep ** 2

        with torch.no_grad():
            for i, atom in enumerate(self.system.atoms):
                atom.position += position_delta[i]
                atom.position.grad = None
        # Invalidate the cached total energy
        self.system.potential_energy = None

        new_acceleration = self._get_acceleration()
        velocity_delta = 0.5 * self.timestep * (acceleration + new_acceleration)

        for i in range(len(self.system.atoms)):
            self.system.velocities[i] += velocity_delta[i]
        # Since only the velocities were updated, the total energy is still valid here
        # Also don't zero out the gradients since they are needed for the next iteration