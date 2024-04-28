import argparse
import importlib.util
import pathlib

from torch.optim import Adam
from matplotlib import pyplot as plt

from molecular_mechanics.integration import VerletIntegrator
from molecular_mechanics.logging import XYZTrajectoryWriter, print_system_state
from molecular_mechanics.system import System


def minimize_energy(system: System, max_iterations: int = 1000):
    positions = [atom.position for atom in system.atoms]
    optimizer = Adam(positions)

    for i in range(max_iterations):
        if i % 100 == 0:
            print(f"Iteration {i}")
            print_system_state(system)
        energy = system.get_potential_energy()
        if energy < 1e-4:
            break
        energy.backward()
        optimizer.step()
        optimizer.zero_grad()

def run_dynamics(system: System, iterations: int, trajectory_file: str, verbose: bool = False, energy_plot_file: str | None = None):
    integrator = VerletIntegrator(system)
    print_freq = 100
    sample_freq = 20
    total_energy = []
    potential_energy = []
    kinetic_energy = []
    xyz_writer = XYZTrajectoryWriter(trajectory_file, system)
    for i in range(iterations):
        if energy_plot_file:
            p = system.get_potential_energy()
            k = system.get_kinetic_energy()
            total_energy.append((p + k).item())
            potential_energy.append(p.item())
            kinetic_energy.append(k.item())
        
        if verbose and i % print_freq == 0:
            print(f"Iteration {i}")
            print_system_state(system)

        if i % sample_freq == 0:
            xyz_writer.write()
        integrator.step()
    xyz_writer.close()

    if energy_plot_file:
        plt.title("Energy")
        plt.plot(total_energy, label="Total Energy")
        plt.plot(potential_energy, label="Potential Energy")
        plt.plot(kinetic_energy, label="Kinetic Energy")
        plt.legend()
        plt.savefig(energy_plot_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("-t", "--temperature", type=float, default=300.0)
    parser.add_argument("-it", "--iterations", type=int, default=1000)
    parser.add_argument("-plt", "--save-energy-plot", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-m", "--minimize-energy", action="store_true")
    args = parser.parse_args()

    infile_path = pathlib.Path(args.input_file)

    if infile_path.suffix == ".py":
        # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
        spec = importlib.util.spec_from_file_location(infile_path.stem, str(infile_path))
        assert spec is not None
        py_molecule = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(py_molecule)

        try:
            atoms = py_molecule.atoms
            connections = py_molecule.connections
            force_field = py_molecule.force_field
        except AttributeError:
            print("Input file must define 'atoms', 'connections', and 'force_field' variables")
            exit(1)
    elif infile_path.suffix == ".pdb":
        raise NotImplementedError("PDB file parsing not implemented yet")
    else:
        print(f"Unknown input file format: '${args.input_file.split('.')[-1]}'")

    
    system = System(atoms, connections, force_field, temperature=args.temperature)

    if args.minimize_energy:
        minimize_energy(system)
    
    run_dynamics(system, args.iterations, args.output_file, verbose=args.verbose, energy_plot_file=args.save_energy_plot)