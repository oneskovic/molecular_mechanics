from molecular_mechanics.forcefield_parser import load_forcefield_vectorized, load_forcefield
from molecular_mechanics.pdb_parser import atoms_and_bonds_from_pdb
from molecular_mechanics.system_fast import SystemFast
from molecular_mechanics.system import System
from molecular_mechanics.integration import VerletIntegrator, FastVerletIntegrator
import torch

def run_system_fast_water(iterations):
    ff_file = 'data/tip3p-custom.xml'
    pdb_file = 'data/water/water_64.pdb'
    force_field_slow = load_forcefield(ff_file)
    atoms, connections = atoms_and_bonds_from_pdb(pdb_file, force_field_slow)
    force_field_fast = load_forcefield_vectorized(ff_file, atoms, connections)
    system_fast = SystemFast(atoms, connections, force_field_fast)
    integrator_fast = FastVerletIntegrator(system_fast)
    for i in range(iterations):
        integrator_fast.step()

def run_system_slow_water(iterations):
    ff_file = 'data/tip3p-custom.xml'
    pdb_file = 'data/water/water_64.pdb'
    force_field_slow = load_forcefield(ff_file)
    atoms, connections = atoms_and_bonds_from_pdb(pdb_file, force_field_slow)
    force_field_fast = load_forcefield_vectorized(ff_file, atoms, connections)
    system_fast = System(atoms, connections, force_field_slow)
    integrator_fast = VerletIntegrator(system_fast)
    for i in range(iterations):
        integrator_fast.step()

def test_system_fast_water(benchmark):
    benchmark(run_system_fast_water, 1)

def test_system_slow_water(benchmark):
    benchmark(run_system_slow_water, 1)


    
