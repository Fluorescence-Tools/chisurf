import numpy as np

import mfm
from mfm.io.pdb import keys, formats, assign_element_to_atom_name


def parse_string_pqr(
        string: str,
        **kwargs
):
    rows = string.splitlines()
    verbose = kwargs.get('verbose', mfm.verbose)
    atoms = np.zeros(len(rows), dtype={'names': keys, 'formats': formats})
    ni = 0
    for line in rows:
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip().upper()
            atoms['i'][ni] = ni
            atoms['chain'][ni] = line[21]
            atoms['atom_name'][ni] = atom_name.upper()
            atoms['res_name'][ni] = line[17:20].strip().upper()
            atoms['res_id'][ni] = line[21:27]
            atoms['atom_id'][ni] = line[6:11]
            atoms['coord'][ni][0] = float(line[30:38].strip())
            atoms['coord'][ni][1] = float(line[38:46].strip())
            atoms['coord'][ni][2] = float(line[46:54].strip())
            atoms['radius'][ni] = float(line[63:70].strip())
            atoms['element'][ni] = assign_element_to_atom_name(atom_name)
            atoms['charge'][ni] = float(line[55:62].strip())
            ni += 1
    atoms = atoms[:ni]
    if verbose:
        print("Number of atoms: %s" % (ni + 1))
    return atoms
