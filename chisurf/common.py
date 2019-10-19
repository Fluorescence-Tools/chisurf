"""
common.py

Common data for all scripts in the SuRF-toolbox. In the common
library all kinds of constants are defined. For instance colors
in plots but also constants as the Avogadros number.
"""

from chisurf.settings import structure_data

quencher = structure_data['Quencher']
quencher_names = quencher.keys()
"""Definition of quenching amino-acids and quenching atoms of quencher"""

MAX_BONDS = structure_data['MAX_BONDS']
"""Dictionary of maximum number of bonds per atom type"""

atom_weights = dict(
    (
        key,
        structure_data["Periodic Table"][key]["Atomic weight"]
    )
    for key in structure_data["Periodic Table"].keys()
)
"""Atomic weights (http://www.chem.qmul.ac.uk/iupac/AtWt/ & PyMol) """

PKA_DICT = structure_data['PKA_DICT']
"""Dictionary of pKa values and un-protonated charge state."""

CHARGE_DICT = structure_data['CHARGE_DICT']
"""Default charges of amino acids"""

TITR_ATOM = structure_data['TITR_ATOM']
"""Atom on which to place charge in amino-acid"""

TITR_ATOM_COARSE = structure_data['TITR_ATOM_COARSE']
"""Atom on which to place charge in amino-acid (Coarse grained default 
position C-Beta)"""

MW_DICT = structure_data['MW_DICT']
"""Dictionary of amino acid molecular weights.  The the molecular weight of 
water should be subtracted for each peptide bond to calculate a protein 
molecular weight."""

MW_H2O = 18.0
"""Molecular weight of water"""

VDW_DICT = dict(
    (key, structure_data["Periodic Table"][key]["vdW radius"])
    for key in structure_data["Periodic Table"].keys()
)
"""Dictionary of van der Waal radii
CR - coarse grained Carbon/Calpha
"""

# --------------------------------------------------------------------------- #
# Amino acid name data
# DON'T CHANGE ORDER!!!
# --------------------------------------------------------------------------- #

_aa_index = [
    ('ALA', 'A'),  # 0
    ('CYS', 'C'),  # 1
    ('ASP', 'D'),  # 2
    ('GLU', 'E'),  # 3
    ('PHE', 'F'),  # 4
    ('GLY', 'G'),  # 5
    ('HIS', 'H'),  # 6
    ('ILE', 'I'),  # 7
    ('LYS', 'K'),  # 8
    ('LEU', 'L'),  # 9
    ('MET', 'M'),  # 10
    ('ASN', 'N'),  # 11
    ('PRO', 'P'),  # 12
    ('GLN', 'Q'),  # 13
    ('ARG', 'R'),  # 14
    ('SER', 'S'),  # 15
    ('THR', 'T'),  # 16
    ('VAL', 'V'),  # 17
    ('TRP', 'W'),  # 18
    ('TYR', 'Y'),  # 19
    ('cisPro', 'cP'),  # 20
    ('transPro', 'tP'),  # 21
    ('CYX', 'C'),  # 22  in Amber CYS with disulfide-bridge
    ('HIE', 'H'),  # 22  in Amber CYS with disulfide-bridge
]

AA3_TO_AA1 = dict(_aa_index)
AA1_TO_AA3 = dict([(aa[1], aa[0]) for aa in _aa_index])
AA3_TO_ID = dict([(aa[0], i) for i, aa in enumerate(_aa_index)])

# --------------------------------------------------------------------------- #
# PDB record data
# --------------------------------------------------------------------------- #

# Types of coordinate entries
COORD_RECORDS = ["ATOM  ", "HETATM"]
