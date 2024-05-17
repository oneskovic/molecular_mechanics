import xml.etree.ElementTree as ET
from molecular_mechanics.forces import (
    HarmonicBondForceParams,
    HarmonicAngleForceParams,
    LennardJonesForceParams,
)
from molecular_mechanics.forces import (
    HarmonicBondForce,
    HarmonicAngleForce,
    LennardJonesForce,
    CoulombForce,
)
from molecular_mechanics.forces_fast import (
    HarmonicBondForceFast,
    HarmonicAngleForceFast,
    LennardJonesForceFast,
    CoulombForceFast
)
from molecular_mechanics.forces import ForceField
from molecular_mechanics.residue_database import load_residue_database
from molecular_mechanics.atom import AtomType, Atom
from molecular_mechanics.forces_fast import ForceFieldVectorized
from molecular_mechanics.molecule import (
    Graph,
    get_all_angles,
    get_all_bonds,
    get_all_dihedrals,
    get_all_pairs_bond_separation,
)



def load_forcefield(forcefield_file: str) -> ForceField:
    tree = ET.parse(forcefield_file)
    atom_types = []
    # Load atom masses
    for child in tree.findall("AtomTypes/Type"):
        element = child.attrib["element"]
        name = child.attrib["name"]
        mass = float(child.attrib["mass"])
        element_class = child.attrib["class"]
        atom_types.append(AtomType(element, name, element_class, mass))

    harmonic_force_dict = dict()
    # Load the harmonic bond force
    for child in tree.findall('HarmonicBondForce/Bond'):
        type1 = child.attrib['type1']
        type2 = child.attrib['type2']
        length = float(child.attrib['length'])
        k = float(child.attrib['k'])
        harmonic_force_dict[(type1,type2)] = HarmonicBondForceParams(length, k)
    
    harmonic_angle_force_dict = dict()
    # Load the harmonic angle force
    for child in tree.findall("HarmonicAngleForce/Angle"):
        type1 = child.attrib["type1"]
        type2 = child.attrib["type2"]
        type3 = child.attrib["type3"]
        angle = float(child.attrib["angle"])
        k = float(child.attrib["k"])
        harmonic_angle_force_dict[(type1, type2, type3)] = HarmonicAngleForceParams(
            angle, k
        )

    lennard_jones_force_dict = dict()
    # Load the lennard jones force
    for child in tree.findall("NonbondedForce/Atom"):
        type = child.attrib["type"]
        sigma = float(child.attrib["sigma"])
        epsilon = float(child.attrib["epsilon"])
        lennard_jones_force_dict[type] = LennardJonesForceParams(epsilon, sigma)

    harmonic_bond_force = HarmonicBondForce(harmonic_force_dict)
    harmonic_angle_force = HarmonicAngleForce(harmonic_angle_force_dict)
    lennard_jones_force = LennardJonesForce(lennard_jones_force_dict)
    coulomb_force = CoulombForce()
    residue_db = load_residue_database(forcefield_file)

    return ForceField(
        atom_types,
        residue_db,
        harmonic_bond_forces=harmonic_bond_force,
        harmonic_angle_forces=harmonic_angle_force,
        lennard_jones_forces=lennard_jones_force,
        coulomb_forces=coulomb_force,
    )

def load_forcefield_vectorized(forcefield_file: str, atoms: list[Atom], connections: Graph,) -> ForceFieldVectorized:
    forcefield = load_forcefield(forcefield_file)
    harmonic_bond_forces_fast = None
    harmonic_angle_forces_fast = None
    lennard_jones_forces_fast = None
    coulomb_forces_fast = None

    if forcefield.harmonic_bond_forces is not None:
        harmonic_bond_forces_fast = HarmonicBondForceFast(
            forcefield.harmonic_bond_forces.bond_dict,
            get_all_bonds(connections),
            atoms
        )
    if forcefield.harmonic_angle_forces is not None:
        harmonic_angle_forces_fast = HarmonicAngleForceFast(
            forcefield.harmonic_angle_forces.angle_dict,
            get_all_angles(connections),
            atoms
        )
    if forcefield.lennard_jones_forces is not None:
        lennard_jones_forces_fast = LennardJonesForceFast(
            forcefield.lennard_jones_forces.lj_dict,
            get_all_pairs_bond_separation(connections),
            atoms,
            forcefield.non_bonded_scaling_factor
        )
    if forcefield.coulomb_forces is not None:
        coulomb_forces_fast = CoulombForceFast(
            get_all_pairs_bond_separation(connections),
            atoms,
            forcefield.non_bonded_scaling_factor
        )
        
    return ForceFieldVectorized(
        atom_types=forcefield.atom_types,
        residue_database=forcefield.residue_database,
        harmonic_bond_forces=harmonic_bond_forces_fast,
        harmonic_angle_forces=harmonic_angle_forces_fast,
        dihedral_forces=forcefield.dihedral_forces,
        lennard_jones_forces=lennard_jones_forces_fast,
        coulomb_forces=coulomb_forces_fast
    )