from typing import TypeGuard

import torch
from torch import Tensor

from molecular_mechanics.system import System
from molecular_mechanics.system_fast import SystemFast

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
            return all(grad is not None and not torch.isnan(grad).any() for grad in grad_list)
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

        new_acceleration = self._get_acceleration()
        velocity_delta = 0.5 * self.timestep * (acceleration + new_acceleration)

        for i in range(len(self.system.atoms)):
            self.system.velocities[i] += velocity_delta[i]
        # Don't zero out the gradients here, as they are needed for the next iteration

class FastVerletIntegrator:
    """
    Verlet integrator for molecular dynamics.

    Using the Velocity Verlet variant without half-steps.
    Described in the Wikipedia article:
    https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
    """

    def __init__(self, system: SystemFast, timestep: float = 0.001):
        self.system = system
        self.timestep = timestep

    def _is_valid_grad(self, grad: Tensor | None) -> TypeGuard[Tensor]:
        return grad is not None and not torch.isnan(grad).any()

    def _get_acceleration(self) -> Tensor:
        potential_energy = self.system.get_potential_energy()
        grad = self.system.atom_positions.grad
        if not self._is_valid_grad(grad):
            potential_energy.backward()
            grad = self.system.atom_positions.grad
            assert self._is_valid_grad(grad)
            
        force = -grad
        acceleration = force / self.system.atom_masses[:, None]
        return acceleration
    
    def step(self) -> None:
        """
        Perform a single iteration of the Verlet integration algorithm.
        """
        acceleration = self._get_acceleration()
        position_delta = self.timestep * self.system.velocities + 0.5 * acceleration * self.timestep ** 2

        with torch.no_grad():
            self.system.atom_positions += position_delta
            self.system.atom_positions.grad = None

        new_acceleration = self._get_acceleration()
        velocity_delta = 0.5 * self.timestep * (acceleration + new_acceleration)

        self.system.velocities += velocity_delta
        # Don't zero out the gradients here, as they are needed for the next iteration
