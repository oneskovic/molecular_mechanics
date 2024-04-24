from Bio.PDB import PDBParser

# Function to extract atom information from a PDB structure
def extract_atoms(structure):
    atom_info = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_name = atom.get_name()
                    residue_name = residue.get_resname()
                    chain_id = chain.get_id()
                    residue_id = residue.get_id()[1]
                    atom_pos = atom.get_coord()
                    atom_type = atom.element
                    atom_key = (chain_id, residue_id, residue_name, atom_name)
                    atom_info[atom_key] = {'position': atom_pos, 'type': atom_type}
    return atom_info

# Function to convert a PDB file to an XYZ file
# Loads a pdb file called <pdb_id>.pdb from the data directory
# and writes the corresponding xyz file to the same directory
def pdb_to_xyz(pdb_id):
    # Load PDB file
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(pdb_id, f'data/{pdb_id}.pdb')

    # Extract atom and connectivity information
    atom_info = extract_atoms(structure)

    with open(f'data/{pdb_id}.xyz', 'w+') as f:
        # Output number of atoms
        f.write(f"{len(atom_info)}\n")
        # Output atom type and position
        for atom_key, info in atom_info.items():
            chain_id, residue_id, residue_name, atom_name = atom_key
            atom_type = info['type']
            atom_pos = info['position']
            f.write(f"{atom_type} {atom_pos[0]} {atom_pos[1]} {atom_pos[2]}\n")

