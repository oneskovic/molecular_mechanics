from typing import Protocol
import sys

import matplotlib.pyplot as plt
from torch import Tensor

from molecular_mechanics.logging import print_system_state
from molecular_mechanics.system import System
from molecular_mechanics.system_fast import SystemFast

from tqdm import tqdm

class Callback(Protocol):
    def __call__(self, i: int, system: System | SystemFast):
        pass
    def close(self):
        pass

class SystemStatePrinting(Callback):
    def __init__(self, printing_freq: int = 100):
        self.printing_freq = printing_freq

    def __call__(self, i, system):
        if i % self.printing_freq == 0:
            print(f"Iteration {i}")
            print_system_state(system)
    
    def close(self):
        pass

class ProgressBar(Callback):
    def __init__(self, max_iterations: int):
        self.pbar = tqdm(total=max_iterations, file=sys.stdout)
        # Save stdout to restore it later and redirect to tqdm.write
        self.save_stdout = sys.stdout
        sys.stdout = self.DummyFile(self.save_stdout)
        self.max_iterations = max_iterations
    
    def __call__(self, i, system):
        self.pbar.update(1)
    
    def close(self):
        self.pbar.close()
        sys.stdout = self.save_stdout
    
    # Dummy file that writes to tqdm.write
    class DummyFile(object):
        file = None
        def __init__(self, file):
            self.file = file

        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x, file=self.file)

class TrajectoryWriting(Callback):
    def __init__(self, trajectory_writer, sample_freq: int = 20):
        self.trajectory_writer = trajectory_writer
        self.sample_freq = sample_freq
    
    def __call__(self, i, system):
        if i % self.sample_freq == 0:
            self.trajectory_writer.write(system)
    
    def close(self):
        self.trajectory_writer.close()

class Plotting(Callback):
    def __init__(self, plot_path: str):
        self.plot_path = plot_path
        self.total_energy = []
        self.potential_energy = []
        self.kinetic_energy = []
        self.iterations = []
        self.fig, self.ax = plt.subplots()
    
    def __call__(self, i: int, system: System):
        p = system.get_potential_energy()
        k = system.get_kinetic_energy()
        self.total_energy.append((p + k).item())
        self.potential_energy.append(p.item())
        self.kinetic_energy.append(k.item())
        self.iterations.append(i)

    def close(self):
        self.ax.set_title("Energy")
        self.ax.plot(self.iterations, self.total_energy, label="Total Energy")
        self.ax.plot(self.iterations, self.potential_energy, label="Potential Energy")
        self.ax.plot(self.iterations, self.kinetic_energy, label="Kinetic Energy")
        self.ax.legend()
        self.fig.savefig(self.plot_path)
        plt.close(self.fig)

class EnergyDiff(Callback):
    def __init__(self, printing_freq: int = 10):
        self.printing_freq = printing_freq
        
        self._energy = None

    def __call__(self, i: int, system: System):
        new_energy = system.get_potential_energy()
        if self._energy is not None and i % self.printing_freq == 0:
            print(f"Energy difference: {abs(new_energy - self._energy)}")
        self._energy = new_energy

    def close(self):
        pass


class EnergyMonitor(Callback):
    def __init__(self) -> None:
        self.potential_energies : list[float] = []
        self.kinetic_energies : list[float] = []

    def __call__(self, _, system: System):
        self.potential_energies.append(system.get_potential_energy().item())
        self.kinetic_energies.append(system.get_kinetic_energy().item())

    def close(self):
        pass
