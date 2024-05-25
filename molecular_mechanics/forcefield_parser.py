import xml.etree.ElementTree as ET
from molecular_mechanics.forces import (
    HarmonicBondForceParams,
    HarmonicAngleForceParams,
    LennardJonesForceParams,
    DihedralForceParams
)
from molecular_mechanics.forces import (
    HarmonicBondForce,
    HarmonicAngleForce,
    LennardJonesForce,
    CoulombForce,
    DihedralForce
)
from molecular_mechanics.forces_fast import (
    HarmonicBondForceFast,
    HarmonicAngleForceFast,
    LennardJonesForceFast,
    CoulombForceFast,
    DihedralForceFast
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
from molecular_mechanics.constants import WILDCARD



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

    # Load the periodic dihedral force
    dihedral_force_dict = dict()
    for child in tree.findall("PeriodicTorsionForce/Proper"):
        type1 = child.attrib["type1"]
        type2 = child.attrib["type2"]
        type3 = child.attrib["type3"]
        type4 = child.attrib["type4"]
        if type1 == "":
            type1 = WILDCARD
        if type4 == "":
            type4 = WILDCARD

        n_terms = max([int(x[-1]) for x in child.attrib if x.startswith('periodicity')])
        ks = []
        ns = []
        phases = []
        for i in range(1, n_terms+1):
            k = float(child.attrib[f"k{i}"])
            n = int(child.attrib[f"periodicity{i}"])
            phase = float(child.attrib[f"phase{i}"])
            ks.append(k)
            ns.append(n)
            phases.append(phase)
        dihedral_force_dict[(type1, type2, type3, type4)] = DihedralForceParams(ks, ns, phases)


    harmonic_bond_force = HarmonicBondForce(harmonic_force_dict)
    harmonic_angle_force = HarmonicAngleForce(harmonic_angle_force_dict)
    coulomb_force = CoulombForce()
    if len(lennard_jones_force_dict) == 0:
        lennard_jones_force = None
    else:
        lennard_jones_force = LennardJonesForce(lennard_jones_force_dict)
    if len(dihedral_force_dict) > 0:
        dihedral_force = DihedralForce(dihedral_force_dict)
    else:
        dihedral_force = None
    residue_db = load_residue_database(forcefield_file)

    return ForceField(
        atom_types,
        residue_db,
        harmonic_bond_forces=harmonic_bond_force,
        harmonic_angle_forces=harmonic_angle_force,
        lennard_jones_forces=lennard_jones_force,
        coulomb_forces=coulomb_force,
        dihedral_forces=dihedral_force
    )

def load_forcefield_vectorized(forcefield_file: str, atoms: list[Atom], connections: Graph,) -> ForceFieldVectorized:
    
    forcefield = load_forcefield(forcefield_file)
    harmonic_bond_forces_fast = None
    harmonic_angle_forces_fast = None
    lennard_jones_forces_fast = None
    coulomb_forces_fast = None
    dihedral_forces_fast = None

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
    if forcefield.dihedral_forces is not None:
        dihedral_forces_fast = DihedralForceFast(
            forcefield.dihedral_forces.dihedral_dict,
            get_all_dihedrals(connections),
            atoms
        )

    return ForceFieldVectorized(
        atom_types=forcefield.atom_types,
        residue_database=forcefield.residue_database,
        harmonic_bond_forces=harmonic_bond_forces_fast,
        harmonic_angle_forces=harmonic_angle_forces_fast,
        dihedral_forces=dihedral_forces_fast,
        lennard_jones_forces=lennard_jones_forces_fast,
        coulomb_forces=coulomb_forces_fast
    )