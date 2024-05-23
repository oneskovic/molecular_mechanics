from molecular_mechanics.atom import Atom
from molecular_mechanics.constants import ANGSTROM2NM
def format_pdb_line(atom_id : int, atom : Atom, molecule_id : int) -> str:
        """
        Example line:
        ATOM   2209  CB  TYR A 299       6.167  22.607  20.046  1.00  8.12           C
        00000000011111111112222222222333333333344444444445555555555666666666677777777778
        12345678901234567890123456789012345678901234567890123456789012345678901234567890

        ATOM line format description from
          http://deposit.rcsb.org/adit/docs/pdb_atom_format.html:

        COLUMNS        DATA TYPE       CONTENTS
        --------------------------------------------------------------------------------
         1 -  6        Record name     "ATOM  "
         7 - 11        Integer         Atom serial number.
        13 - 16        Atom            Atom name.
        17             Character       Alternate location indicator.
        18 - 20        Residue name    Residue name.
        22             Character       Chain identifier.
        23 - 26        Integer         Residue sequence number.
        27             AChar           Code for insertion of residues.
        31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
        39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
        47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
        55 - 60        Real(6.2)       Occupancy (Default = 1.0).
        61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
        73 - 76        LString(4)      Segment identifier, left-justified.
        77 - 78        LString(2)      Element symbol, right-justified.
        79 - 80        LString(2)      Charge on the atom.

        """
        position = atom.position.clone()
        position /= ANGSTROM2NM
        line = list(" "*80)
        line[0:6] = f"{'ATOM':<6}"
        line[6:11] = f"{atom_id:>5}"
        line[12:16] = f"{atom.element:<4}"
        line[17:20] = f"{atom.residue:>3}"
        line[21] = "A"
        line[22:26] = f"{molecule_id:>4}"
        line[30:38] = f"{position[0]:>8.3f}"
        line[38:46] = f"{position[1]:>8.3f}"
        line[46:54] = f"{position[2]:>8.3f}"
        line[54:60] = f"{1.00:>6.2f}"
        line[60:66] = f"{0.00:>6.2f}"
        line[76:78] = f"{atom.atom_type.element:>2}"
        #return f"{'ATOM':<6}{atom_id:>5} {atom.element:>4} {'HOH':>3} A{molecule_id:>4} {atom.position[0]:>8.3f}{atom.position[1]:>8.3f}{atom.position[2]:>8.3f}{1.00:>6.2f}{0.00:>6.2f}{' ':>9}{atom.atom_type.element:>2}  \n"
        return "".join(line) + "\n"
