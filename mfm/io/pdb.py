from __future__ import annotations

import os
import urllib.request
import numpy as np

import mfm
import mfm.common as common

keys_formats = [
    ('i4', 'i'),
    ('|U1', 'chain'),
    ('i4', 'res_id'),
    ('|U5', 'res_name'),
    ('i4', 'atom_id'),
    ('|U5', 'atom_name'),
    ('|U1', 'element'),
    ('3f8', 'coord'),
    ('f8', 'charge'),
    ('f8', 'radius'),
    ('f8', 'bfactor'),
    ('f8', 'mass')
]

keys, formats = list(zip(*keys_formats))


def fetch_pdb_string(
        pdb_id: str
):
    url = 'http://www.rcsb.org/pdb/files/%s.pdb' % pdb_id[:4]
    return urllib.request.urlopen(url).read()


def fetch_pdb(
        pdb_id: str,
        **kwargs
):
    st = fetch_pdb_string(pdb_id)
    return parse_string_pdb(st, **kwargs)


def assign_element_to_atom_name(
        atom_name: str
):
    """Tries to guess element from atom name if not recognised.

    :param atom_name: string

    Examples
    --------

    >>> assign_element_to_atom_name('CA')
    C
    """
    element = ""
    if atom_name.upper() not in common.atom_weights:
        # Inorganic elements have their name shifted left by one position
        #  (is a convention in PDB, but not part of the standard).
        # isdigit() check on last two characters to avoid mis-assignment of
        # hydrogens atoms (GLN HE21 for example)
        # Hs may have digit in [0]
        putative_element = atom_name[1] if atom_name[0].isdigit() else atom_name[0]
        if putative_element.capitalize() in common.atom_weights.keys():
            element = putative_element
    return element


def parse_string_pdb(
        string: str,
        assign_charge: bool = False,
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
            atoms['res_name'][ni] = line[17:20].strip().upper()
            atoms['atom_name'][ni] = atom_name
            atoms['res_id'][ni] = line[22:26]
            atoms['atom_id'][ni] = line[6:11]
            atoms['coord'][ni][0] = line[30:38]
            atoms['coord'][ni][1] = line[38:46]
            atoms['coord'][ni][2] = line[46:54]
            atoms['bfactor'][ni] = line[60:65]
            atoms['element'][ni] = assign_element_to_atom_name(atom_name)
            try:
                if assign_charge:
                    if atoms['res_name'][ni] in common.CHARGE_DICT:
                        if atoms['atom_name'][ni] == common.TITR_ATOM_COARSE[atoms['res_name'][ni]]:
                            atoms['charge'][ni] = common.CHARGE_DICT[atoms['res_name'][ni]]
                atoms['mass'][ni] = common.atom_weights[atoms['element'][ni]]
                atoms['radius'][ni] = common.VDW_DICT[atoms['element'][ni]]
            except KeyError:
                print("Cloud not assign parameters to: %s" % line)
            ni += 1
    atoms = atoms[:ni]
    if verbose:
        print("Number of atoms: %s" % (ni + 1))
    return atoms


def read(
        filename: str,
        assign_charge: bool = False,
        **kwargs
):
    """ Open pdb_file and read each line into pdb (a list of lines)

    :param filename:
    :param assign_charge:
    :return:
        numpy structured array containing the PDB info and VdW-radii and charges

    Examples
    --------

    >>> import mfm
    >>> pdb_file = models'
    >>> pdb = mfm.io.pdb_file.read(pdb_file, verbose=True)
    >>> pdb[:5]
    array([ (0, ' ', 7, 'MET', 1, 'N', 'N', [72.739, -17.501, 8.879], 0.0, 1.65, 0.0, 14.0067),
           (1, ' ', 7, 'MET', 2, 'CA', 'C', [73.841, -17.042, 9.747], 0.0, 1.76, 0.0, 12.0107),
           (2, ' ', 7, 'MET', 3, 'C', 'C', [74.361, -18.178, 10.643], 0.0, 1.76, 0.0, 12.0107),
           (3, ' ', 7, 'MET', 4, 'O', 'O', [73.642, -18.708, 11.489], 0.0, 1.4, 0.0, 15.9994),
           (4, ' ', 7, 'MET', 5, 'CB', 'C', [73.384, -15.89, 10.649], 0.0, 1.76, 0.0, 12.0107)],
          dtype=[('i', '<i4'), ('chain', 'S1'), ('res_id', '<i4'), ('res_name', 'S5'), ('atom_id', '<i4'), ('atom_name', 'S5
    '), ('element', 'S1'), ('coord', '<f8', (3,)), ('charge', '<f8'), ('radius', '<f8'), ('bfactor', '<f8'), ('mass', '<f8')
    ])
    """
    verbose = kwargs.get('verbose', mfm.verbose)
    with open(filename, 'r') as f:
        string = f.read()
        if verbose:
            path, baseName = os.path.split(filename)
            print("======================================")
            print("Filename: %s" % filename)
            print("Path: %s" % path)
        if filename.endswith('.pdb'):
            atoms = parse_string_pdb(string, assign_charge, **kwargs)
        elif filename.endswith('.pqr'):
            atoms = parse_string_pqr(string, **kwargs)
    return atoms


def write_pdb(
        filename: str,
        atoms=None,
        append_model: bool = False,
        append_coordinates: bool = False,
        verbose: bool = False
):
    """ Writes a structured numpy array containing the PDB-info to a PDB-file

    If append_model and append_coordinates are False the file is overwritten. Otherwise the atomic-coordinates
    are appended to the existing file.


    :param filename: target-filename
    :param atoms: structured numpy array
    :param append_model: bool
        If True the atoms are appended as a new models
    :param append_coordinates:
        If True the coordinates are appended to the file

    """
    mode = 'a+' if append_model or append_coordinates else 'w+'
    if verbose:
        print("Writing to file: ", filename)
    with open(filename, mode) as fp:
        # http://cupnet.net/pdb_format/
        al = [
            "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" %
            ("ATOM ", at['atom_id'], at['atom_name'], " ", at['res_name'], at['chain'], at['res_id'], " ",
             at['coord'][0], at['coord'][1], at['coord'][2], 0.0, at['bfactor'], at['element'], "  ")
            for at in atoms
        ]
        if append_model:
            fp.write('MODEL')
        fp.write("".join(al))
        if append_model:
            fp.write('ENDMDL')


def write_points(
        filename: str,
        points,
        verbose: bool = False,
        mode='xyz',
        density=None
):
    """

    :param filename:
    :param points:
    :param verbose:
    :param mode:
    :param density:
    :return:
    """
    if mode == 'pdb':
        atoms = np.empty(len(points), dtype={'names': keys, 'formats': formats})
        atoms['coord'] = points
        if density is not None:
            atoms['bfactor'] = density
        write_pdb(filename, atoms, verbose=verbose)
    else:
        write_xyz(filename, points, verbose=verbose)


def get_atom_index(
        atoms: np.array,
        chain_identifier: str,
        residue_seq_number: int,
        atom_name: str,
        residue_name: str,
        **kwargs
):
    """
    Get the atom index by the the identifier

    :param atoms:
    :param chain_identifier:
    :param residue_seq_number:
    :param atom_name:
    :param residue_name:
    :param kwargs:
    :return:
    """
    # Determine Labeling position
    if residue_seq_number is None or atom_name is None:
        raise ValueError("Either attachment_atom number or residue and atom_name have to be provided.")
    verbose = kwargs.get('verbose', mfm.verbose)
    ignore_multiple_selections = kwargs.get('ignore_multiple_selections', True)
    if verbose:
        print("Labeling position")
        print("Chain ID: %s" % chain_identifier)
        print("Residue seq. number: %s" % residue_seq_number)
        print("Residue name: %s" % residue_name)
        print("Atom name: %s" % atom_name)

    if chain_identifier is None or chain_identifier == '':
        attachment_atom_index = np.where((atoms['res_id'] == residue_seq_number) &
                                         (atoms['atom_name'] == atom_name))[0]
    else:
        attachment_atom_index = np.where((atoms['res_id'] == residue_seq_number) &
                                         (atoms['atom_name'] == atom_name) &
                                         (atoms['chain'] == chain_identifier))[0]
    if len(attachment_atom_index) != 1 and not ignore_multiple_selections:
        print("Labeling position")
        print("Chain ID: %s" % chain_identifier)
        print("Residue seq. number: %s" % residue_seq_number)
        print("Residue name: %s" % residue_name)
        print("Atom name: %s" % atom_name)
        raise ValueError("Invalid selection of attachment atom")
    else:
        attachment_atom_index = attachment_atom_index[0]
    if verbose:
        print("Atom index: %s" % attachment_atom_index)
    return attachment_atom_index


