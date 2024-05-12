from typing import Protocol

import matplotlib.pyplot as plt

from molecular_mechanics.logging import print_system_state
from molecular_mechanics.system import System

from tqdm import tqdm

class Callback(Protocol):
    def __call__(self, i: int, system: System):
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
        self.max_iterations = max_iterations
        self.pbar = tqdm(total=max_iterations)
    
    def __call__(self, i, system):
        self.pbar.update(1)
        # Reset the progress bar if this is the last iteration
        if i == self.max_iterations - 1:
            self.pbar.close()
            self.pbar = tqdm(total=self.max_iterations)
    
    def close(self):
        self.pbar.close()
    

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
