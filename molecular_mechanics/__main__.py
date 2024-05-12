import argparse
import importlib.util
import pathlib

from molecular_mechanics.callbacks import Callback, Plotting, SystemStatePrinting, TrajectoryWriting, ProgressBar
from molecular_mechanics.energy_minimization import minimize_energy
from molecular_mechanics.logging import XYZTrajectoryWriter
from molecular_mechanics.molecular_dynamics import run_dynamics
from molecular_mechanics.system import System
from molecular_mechanics.forcefield_parser import load_forcefield
from molecular_mechanics.pdb_parser import atoms_and_bonds_from_pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("-t", "--temperature", type=float)
    parser.add_argument("-it", "--iterations", type=int, default=1000)
    parser.add_argument("-plt", "--save-energy-plot", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-m", "--minimize-energy", action="store_true")
    parser.add_argument("-mit", "--minimize-iterations", type=int, default=1000)
    parser.add_argument("-ff", "--force-field", type=str, default="data/ff14SB.xml")
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
        force_field = load_forcefield(args.force_field)
        atoms, connections = atoms_and_bonds_from_pdb(str(infile_path), force_field)
    else:
        print(f"Unknown input file format: '${args.input_file.split('.')[-1]}'")
        exit(1)

    
    system = System(atoms, connections, force_field, temperature=args.temperature) 
        
    class DynamicsCallback(Callback):
        def __init__(self):
            self.callbacks = []
            if args.verbose:
                self.callbacks.append(SystemStatePrinting())
            if args.save_energy_plot:
                self.callbacks.append(Plotting(args.save_energy_plot))
            self.callbacks.append(ProgressBar(args.iterations))
            self.trajectory_writer = XYZTrajectoryWriter(args.output_file)
            self.callbacks.append(TrajectoryWriting(self.trajectory_writer))
        
        def __call__(self, i: int, system: System):
            for callback in self.callbacks:
                callback(i, system)
        
        def close(self):
            for callback in self.callbacks:
                callback.close()
            self.trajectory_writer.close()

    if args.minimize_energy:
        minimize_energy(system, max_iterations=args.minimize_iterations)
    run_dynamics(system, args.iterations, DynamicsCallback()) 