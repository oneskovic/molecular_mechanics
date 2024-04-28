import xml.etree.ElementTree as ET
from molecular_mechanics.forces import HarmonicBondForceParams, HarmonicAngleForceParams, LennardJonesForceParams
from molecular_mechanics.forces import HarmonicBondForce, HarmonicAngleForce, LennardJonesForce, CoulombForce
from molecular_mechanics.forces import ForceField
from molecular_mechanics.residue_database import load_residue_database
from molecular_mechanics.atom import AtomType

def load_forcefield(forcefield_file = 'data/ff14SB.xml') -> ForceField:
    tree = ET.parse(forcefield_file)
    atom_types = []
    # Load atom masses
    for child in tree.findall('AtomTypes/Type'):
        element = child.attrib['element']
        name = child.attrib['name']
        mass = float(child.attrib['mass'])
        element_class = child.attrib['class']
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
    for child in tree.findall('HarmonicAngleForce/Angle'):
        type1 = child.attrib['type1']
        type2 = child.attrib['type2']
        type3 = child.attrib['type3']
        angle = float(child.attrib['angle'])
        k = float(child.attrib['k'])
        harmonic_angle_force_dict[(type1,type2,type3)] = HarmonicAngleForceParams(angle,k)
    
    lennard_jones_force_dict = dict()
    # Load the lennard jones force
    for child in tree.findall('NonbondedForce/Atom'):
        type = child.attrib['type']
        sigma = float(child.attrib['sigma'])
        epsilon = float(child.attrib['epsilon'])
        lennard_jones_force_dict[type] = LennardJonesForceParams(epsilon,sigma)
        
    harmonic_bond_force = HarmonicBondForce(harmonic_force_dict)
    harmonic_angle_force = HarmonicAngleForce(harmonic_angle_force_dict)
    lennard_jones_force = LennardJonesForce(lennard_jones_force_dict)
    coulomb_force = CoulombForce()
    residue_db = load_residue_database(forcefield_file)

    return ForceField(residue_db, atom_types, 
                      harmonic_bond_forces=harmonic_bond_force, 
                      harmonic_angle_forces=harmonic_angle_force, 
                      lennard_jones_forces=lennard_jones_force,
                      coulomb_forces=coulomb_force)

