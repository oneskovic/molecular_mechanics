import xml.etree.ElementTree as ET
from molecular_mechanics.atom import Atom

class Residue:
    def __init__(self, name: str, atom_charges: dict[str, float], bonds : list[tuple[str, str]]):
        self.name = name
        self.atom_charges = atom_charges
        self.bonds = bonds

class ResidueDatabase:
    def __init__(self, residues: dict[str, Residue]):
        self.residues = residues
    
    def get_charge(self, residue, atom_name):
        return self.residues[residue].atom_charges[atom_name]
        
def ff14sb_residue_database(forcefield_file = 'data/tip3p.xml') -> ResidueDatabase:
    tree = ET.parse(forcefield_file)
    residues = {}
    for residue in tree.findall('Residues/Residue'):
        name = residue.attrib['name']
        atom_charges = {}
        bonds = []
        for element in residue:
            if element.tag == 'Atom':
                atom_name = element.attrib['name']
                atom_charge = float(element.attrib['charge'])
                atom_type = element.attrib['type']
                atom_charges[atom_name] = atom_charge
            elif element.tag == 'Bond':
                atom1 = element.attrib['atomName1']
                atom2 = element.attrib['atomName2']
                bonds.append((atom1, atom2))
        residues[name] = Residue(name, atom_charges, bonds)
    return ResidueDatabase(residues)