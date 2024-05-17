from molecular_mechanics.forcefield_parser import load_forcefield_vectorized, load_forcefield
from molecular_mechanics.pdb_parser import atoms_and_bonds_from_pdb
from molecular_mechanics.system_fast import SystemFast
from molecular_mechanics.system import System
from molecular_mechanics.integration import VerletIntegrator, FastVerletIntegrator
import molecular_mechanics.config
import torch
import os

def run_test(ff_file, pdb_file, iterations):
    molecular_mechanics.config.TORCH_DEVICE = "cpu"
    # Dirty hack :(
    force_field_slow = load_forcefield(ff_file)
    atoms, connections = atoms_and_bonds_from_pdb(pdb_file, force_field_slow)
    force_field_fast = load_forcefield_vectorized(ff_file, atoms, connections)
    system_fast = SystemFast(atoms, connections, force_field_fast)
    system_slow = System(atoms, connections, force_field_slow)

    integrator_slow = VerletIntegrator(system_slow)
    integrator_fast = FastVerletIntegrator(system_fast)
    for i in range(iterations):
        integrator_slow.step()
        integrator_fast.step()
        with torch.no_grad():
            atoms_slow_tensor = torch.stack([atom.position for atom in system_slow.atoms]).cpu()
            atoms_fast_tensor = system_fast.atom_positions
            test_ok = torch.allclose(atoms_slow_tensor, atoms_fast_tensor, atol=0.01, rtol=0.01)
            if not test_ok:
                print(atoms_slow_tensor)
                print(atoms_fast_tensor)
            assert test_ok

def test_system_fast_water():
    ff_file = 'data/tip3p.xml'
    pdb_file = 'data/water/water_five.pdb'
    run_test(ff_file, pdb_file, 250)

def test_system_fast_protein():
    ff_file = 'data/ff14SB.xml'
    pdb_file = 'data/proteins/1cfg.pdb'
    run_test(ff_file, pdb_file, 1)
