"""Read PDB files.

PDB files contain atomic coordinates.

`PDB <http://www.wwpdb.org/documentation/file-format>`_

:Author:
  `Thomas-Otavio Peulen <http://tpeulen.github.io>`_

Requirements
------------


Revisions
---------

Notes
-----
The API is not stable yet and might change between revisions.

References
----------

Examples
--------

"""

from __future__ import annotations

import os
import urllib.request
import numpy as np

import scikit_fluorescence as skf
import scikit_fluorescence.io.zipped
import scikit_fluorescence.io.structure

import chisurf
import chisurf.common


write_pdb = skf.io.structure.write_pdb

keys_formats = [
    ('i', 'i4'),
    ('chain', '|U1'),
    ('res_id', 'i4'),
    ('res_name', '|U5'),
    ('atom_id', 'i4'),
    ('atom_name', '|U5'),
    ('element', '|U1'),
    ('xyz', '3f8'),
    ('charge', 'f8'),
    ('radius', 'f8'),
    ('bfactor', 'f8'),
    ('mass', 'f8')
]

keys, formats = list(zip(*keys_formats))
get_atom_index = skf.io.structure.find_atom_index


def fetch_pdb_string(
        pdb_id: str
) -> str:
    """Downloads from the RCSB a PDB file with the specified PDB-ID

    :param pdb_id: The PDB-ID that is downloaded
    :param get_binary: If get_binary is True a binary string is returned.
    :return:
    """
    url = 'http://www.rcsb.org/pdb/files/%s.pdb' % pdb_id[:4]
    binary = urllib.request.urlopen(url).read()
    return binary.decode("utf-8")


def fetch_pdb(pdb_id: str, **kwargs):
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
    element = atom_name
    if atom_name.upper() not in chisurf.common.atom_weights:
        # Inorganic elements have their name shifted left by one position
        #  (is a convention in PDB, but not part of the standard).
        # isdigit() check on last two characters to avoid mis-assignment of
        # hydrogens atoms (GLN HE21 for example)
        # Hs may have digit in [0]
        putative_element = atom_name[1] if atom_name[0].isdigit() else \
            atom_name[0]
        if putative_element.capitalize() in chisurf.common.atom_weights.keys():
            element = putative_element
    return element


def parse_string_pdb(
        string: str,
        assign_charge: bool = False,
        verbose: bool = chisurf.verbose
):
    """

    :param string:
    :param assign_charge:
    :param verbose:
    :return:
    """
    rows = string.splitlines()
    atoms = np.zeros(
        len(rows),
        dtype={
            'names': keys,
            'formats': formats
        }
    )
    ni = 0
    for line in rows:
        if verbose:
            print(line)
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip().upper()
            atoms['i'][ni] = ni
            atoms['chain'][ni] = line[21]
            atoms['res_name'][ni] = line[17:20].strip().upper()
            atoms['atom_name'][ni] = atom_name
            atoms['res_id'][ni] = line[22:26]
            atoms['atom_id'][ni] = line[6:11]
            atoms['xyz'][ni][0] = line[30:38]
            atoms['xyz'][ni][1] = line[38:46]
            atoms['xyz'][ni][2] = line[46:54]
            atoms['bfactor'][ni] = line[60:65]
            atoms['element'][ni] = assign_element_to_atom_name(atom_name)
            try:
                if assign_charge:
                    if atoms['res_name'][ni] in chisurf.common.CHARGE_DICT:
                        if atoms['atom_name'][ni] == chisurf.common.TITR_ATOM_COARSE[atoms['res_name'][ni]]:
                            atoms['charge'][ni] = chisurf.common.CHARGE_DICT[
                                atoms['res_name'][ni]
                            ]
                atoms['mass'][ni] = chisurf.common.atom_weights[atoms['element'][ni]]
                atoms['radius'][ni] = chisurf.common.VDW_DICT[atoms['element'][ni]]
            except KeyError:
                print("Cloud not assign parameters to: %s" % line)
            ni += 1
    atoms = atoms[:ni]
    if verbose:
        print("Number of atoms: %s" % (ni + 1))
    return atoms


def parse_string_pqr(
        string: str,
        verbose: bool = chisurf.verbose
):
    rows = string.splitlines()
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
            atoms['xyz'][ni][0] = float(line[30:38].strip())
            atoms['xyz'][ni][1] = float(line[38:46].strip())
            atoms['xyz'][ni][2] = float(line[46:54].strip())
            atoms['radius'][ni] = float(line[63:70].strip())
            atoms['element'][ni] = assign_element_to_atom_name(atom_name)
            atoms['charge'][ni] = float(line[55:62].strip())
            atoms['element'][ni] = assign_element_to_atom_name(atom_name)
            try:
                atoms['mass'][ni] = chisurf.common.atom_weights[atoms['element'][ni]]
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
        verbose: bool = chisurf.verbose,
        **kwargs
) -> np.ndarray:
    """ Open pdb_file and read each line into pdb (a list of lines)

    :param filename:
    :param assign_charge:
    :return:
        numpy structured array containing the PDB info and VdW-radii and charges

    Examples
    --------

    >>> import chisurf.fio
    >>> pdb_file = './test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb'
    >>> pdb = chisurf.fio.structure.coordinates.read(pdb_file, verbose=True)
    >>> pdb[:5]
    array([ (0, ' ', 7, 'MET', 1, 'N', 'N', [72.739, -17.501, 8.879], 0.0, 1.65, 0.0, 14.0067),
           (1, ' ', 7, 'MET', 2, 'CA', 'C', [73.841, -17.042, 9.747], 0.0, 1.76, 0.0, 12.0107),
           (2, ' ', 7, 'MET', 3, 'C', 'C', [74.361, -18.178, 10.643], 0.0, 1.76, 0.0, 12.0107),
           (3, ' ', 7, 'MET', 4, 'O', 'O', [73.642, -18.708, 11.489], 0.0, 1.4, 0.0, 15.9994),
           (4, ' ', 7, 'MET', 5, 'CB', 'C', [73.384, -15.89, 10.649], 0.0, 1.76, 0.0, 12.0107)],
          dtype=[('i', '<i4'), ('chain', 'S1'), ('res_id', '<i4'), ('res_name', 'S5'), ('atom_id', '<i4'), ('atom_name', 'S5
    '), ('element', 'S1'), ('xyz', '<f8', (3,)), ('charge', '<f8'), ('radius', '<f8'), ('bfactor', '<f8'), ('mass', '<f8')
    ])
    """
    if os.path.isfile(filename):
        with scikit_fluorescence.io.zipped.open_maybe_zipped(
                filename=filename,
                mode='r'
        ) as f:
            string = f.read()
            if verbose:
                path, baseName = os.path.split(filename)
                print("======================================")
                print("Filename: %s" % filename)
                print("Path: %s" % path)
            fn1, ext1 = os.path.splitext(filename.upper())
            _, ext2 = os.path.splitext(fn1)
            # # PDB, mmCIF now handled by scikit_fluorescence
            # if '.PDB' in [ext1, ext2]:
            #     atoms = parse_string_pdb(string, assign_charge, **kwargs)
            if '.PQR' in [ext1, ext2]:
                atoms = parse_string_pqr(string, **kwargs)
            else:
                atoms = skf.io.structure.read_coordinates(
                    filename=filename
                )
            return atoms
    else:
        return np.zeros(0, dtype={'names': keys, 'formats': formats})
