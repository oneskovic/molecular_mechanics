import numpy as np

from molecular_mechanics.atom import Atom
from molecular_mechanics.callbacks import EnergyMonitor
from molecular_mechanics.forces import ForceField
from molecular_mechanics.molecular_dynamics import run_dynamics
from molecular_mechanics.molecule import Graph
from molecular_mechanics.system import System
from molecular_mechanics.forces_fast import HarmonicBondForceFast, CoulombForceFast, HarmonicAngleForceFast, LennardJonesForceFast
import torch
import molecular_mechanics.config as conf

def assert_percentage_diff(a, b, threshold=0.1):
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    abs_diff = np.abs(a - b)
    percentage_diff = abs_diff / a
    assert percentage_diff < threshold

def total_energies_on_molecule(molecule: tuple[list[Atom], ForceField, Graph]):
        atoms, forcefield, connections = molecule
        system = System(atoms, connections, forcefield)
        for _ in range(100):
            with torch.no_grad():
                atom_positions = torch.stack([atom.position for atom in atoms]).to(conf.TORCH_DEVICE)
                if forcefield.harmonic_bond_forces is not None:
                    # Test HarmonicBondForce
                    bonds_energy_slow = system.get_bonds_energy()
                    fast_bond_force = HarmonicBondForceFast(forcefield.harmonic_bond_forces.bond_dict, system.bonds, atoms)
                    bonds_energy_fast = fast_bond_force.get_forces(atom_positions)
                    assert_percentage_diff(bonds_energy_slow, bonds_energy_fast)

                if forcefield.lennard_jones_forces is not None:
                    # Test CoulombForce + LennardJonesForce
                    nonbonded_energy_slow = system.get_non_bonded_energy()
                    fast_coulomb_force = CoulombForceFast(system.all_pairs_bond_separation, atoms, forcefield.non_bonded_scaling_factor)
                    fast_lennard_jones_force = LennardJonesForceFast(forcefield.lennard_jones_forces.lj_dict, system.all_pairs_bond_separation, atoms, forcefield.non_bonded_scaling_factor)
                    nonbonded_energy_fast = fast_coulomb_force.get_forces(atom_positions) + fast_lennard_jones_force.get_forces(atom_positions)
                    assert_percentage_diff(nonbonded_energy_slow, nonbonded_energy_fast)

                if forcefield.harmonic_angle_forces is not None:
                    # Test HarmonicAngleForce
                    angles_energy_slow = system.get_angles_energy()
                    fast_angle_force = HarmonicAngleForceFast(forcefield.harmonic_angle_forces.angle_dict, system.angles, atoms)
                    angles_energy_fast = fast_angle_force.get_forces(atom_positions)
                    assert_percentage_diff(angles_energy_slow, angles_energy_fast)
            run_dynamics(system, 1)

def test_three_waters(three_waters):
    total_energies_on_molecule(three_waters)

def test_ethane(ethane):
    total_energies_on_molecule(ethane)