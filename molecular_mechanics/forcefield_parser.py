import xml.etree.ElementTree as ET
from molecular_mechanics.forces import HarmonicBondForceParams, HarmonicAngleForceParams, LennardJonesForceParams
from molecular_mechanics.forces import HarmonicBondForce, HarmonicAngleForce, LennardJonesForce, CoulombForce
from molecular_mechanics.forces import ForceField
from molecular_mechanics.residue_database import ff14sb_residue_database

def ff14sb_forcefield(forcefield_file = 'data/ff14SB.xml') -> ForceField:
    tree = ET.parse(forcefield_file)
    root = tree.getroot()
    masses = dict()
    # Load atom masses
    for child in root[1]:
        element_name = child.attrib['name']
        element_mass = child.attrib['mass']
        masses[element_name] = float(element_mass)
    
    harmonic_force_dict = dict()
    # Load the harmonic bond force
    for child in root[3]:
        type1 = child.attrib['type1']
        type2 = child.attrib['type2']
        length = float(child.attrib['length'])
        k = float(child.attrib['k'])
        harmonic_force_dict[(type1,type2)] = HarmonicBondForceParams(length, k)
    
    harmonic_angle_force_dict = dict()
    # Load the harmonic angle force
    for child in root[4]:
        type1 = child.attrib['type1']
        type2 = child.attrib['type2']
        type3 = child.attrib['type3']
        angle = float(child.attrib['angle'])
        k = float(child.attrib['k'])
        harmonic_angle_force_dict[(type1,type2,type3)] = HarmonicAngleForceParams(angle,k)
    
    lennard_jones_force_dict = dict()
    # Load the lennard jones force
    for child in root[6][1:]:
        type = child.attrib['type']
        sigma = float(child.attrib['sigma'])
        epsilon = float(child.attrib['epsilon'])
        lennard_jones_force_dict[type] = LennardJonesForceParams(epsilon,sigma)
        
    harmonic_bond_force = HarmonicBondForce(harmonic_force_dict)
    harmonic_angle_force = HarmonicAngleForce(harmonic_angle_force_dict)
    lennard_jones_force = LennardJonesForce(lennard_jones_force_dict)
    coulomb_force = CoulombForce()
    residue_db = ff14sb_residue_database(forcefield_file)

    return ForceField(residue_db, masses, 
                      harmonic_bond_forces=harmonic_bond_force, 
                      harmonic_angle_forces=harmonic_angle_force, 
                      lennard_jones_forces=lennard_jones_force,
                      coulomb_forces=coulomb_force)

