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

# Removed for now because mmcif does not exist for Windows
#import mmcif.fio.PdbxReader

import mfm
import chisurf
import chisurf.common as common

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
    element = atom_name
    if atom_name.upper() not in common.atom_weights:
        # Inorganic elements have their name shifted left by one position
        #  (is a convention in PDB, but not part of the standard).
        # isdigit() check on last two characters to avoid mis-assignment of
        # hydrogens atoms (GLN HE21 for example)
        # Hs may have digit in [0]
        putative_element = atom_name[1] if atom_name[0].isdigit() else \
            atom_name[0]
        if putative_element.capitalize() in common.atom_weights.keys():
            element = putative_element
    return element


def parse_string_pdb(
        string: str,
        assign_charge: bool = False,
        verbose: bool = mfm.verbose
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
                    if atoms['res_name'][ni] in common.CHARGE_DICT:
                        if atoms['atom_name'][ni] == common.TITR_ATOM_COARSE[atoms['res_name'][ni]]:
                            atoms['charge'][ni] = common.CHARGE_DICT[
                                atoms['res_name'][ni]
                            ]
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
        verbose: bool = mfm.verbose,
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
    >>> pdb_file = models
    >>> pdb = chisurf.fio.coordinates.read(pdb_file, verbose=True)
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
    with chisurf.fio.zipped.open_maybe_zipped(
            filename=filename,
            mode='r'
    ) as f:
        string = f.read()
        if verbose:
            path, baseName = os.path.split(filename)
            print("======================================")
            print("Filename: %s" % filename)
            print("Path: %s" % path)
        fn1, ext1 = os.path.splitext(filename)
        _, ext2 = os.path.splitext(fn1)
        if '.pdb' in [ext1, ext2]:
            atoms = parse_string_pdb(string, assign_charge, **kwargs)
        else:  # elif filename.endswith('.pqr'):
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
    mode = 'a+' if append_model or append_coordinates else 'w'
    if verbose:
        print("Writing to file: ", filename)
    with chisurf.fio.zipped.open_maybe_zipped(
            filename=filename,
            mode=mode
    ) as fp:
        # http://cupnet.net/pdb_format/
        al = [
            "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" %
            (
                "ATOM ", at['atom_id'], at['atom_name'], " ", at['res_name'],
                at['chain'], at['res_id'], " ",
                at['xyz'][0], at['xyz'][1], at['xyz'][2], 0.0, at['bfactor'],
                at['element'], "  "
            )
            for at in atoms
        ]
        if append_model:
            fp.write('MODEL')
        fp.write("".join(al))
        if append_model:
            fp.write('ENDMDL')


def write_points(
        filename: str,
        points: np.ndarray,
        verbose: bool = False,
        mode: str = 'xyz',
        density: np.ndarray = None
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
        atoms = np.empty(
            len(points),
            dtype={
                'names': keys,
                'formats': formats
            }
        )
        atoms['xyz'] = points
        if density is not None:
            atoms['bfactor'] = density
        write_pdb(
            filename,
            atoms,
            verbose=verbose
        )
    else:
        write_xyz(
            filename,
            points,
            verbose=verbose
        )


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
        raise ValueError(
            "Either attachment_atom number or residue and atom_name have to be provided."
        )
    verbose = kwargs.get('verbose', mfm.verbose)
    ignore_multiple_selections = kwargs.get('ignore_multiple_selections', True)
    if verbose:
        print("Labeling position")
        print("Chain ID: %s" % chain_identifier)
        print("Residue seq. number: %s" % residue_seq_number)
        print("Residue name: %s" % residue_name)
        print("Atom name: %s" % atom_name)

    if chain_identifier is None or chain_identifier == '':
        attachment_atom_index = np.where(
            (atoms['res_id'] == residue_seq_number) &
            (atoms['atom_name'] == atom_name)
        )[0]
    else:
        attachment_atom_index = np.where(
            (atoms['res_id'] == residue_seq_number) &
            (atoms['atom_name'] == atom_name) &
            (atoms['chain'] == chain_identifier)
        )[0]
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


def parse_string_pqr(
        string: str,
        verbose: bool = mfm.verbose
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
            ni += 1
    atoms = atoms[:ni]
    if verbose:
        print("Number of atoms: %s" % (ni + 1))
    return atoms


def write_xyz(
        filename: str,
        points: np.array,
        verbose: bool = mfm.verbose
):
    """
    Writes the points as xyz-format file. The xyz-format file can be opened and displayed for instance
    in PyMol

    :param filename: string
    :param points: array
    :param verbose: bool

    """
    if verbose:
        print("write_xyz\n")
        print("Filename: %s\n" % filename)
    with chisurf.fio.zipped.open_maybe_zipped(
            filename=filename,
            mode='w'
    ) as fp:
        npoints = len(points)
        fp.write('%i\n' % npoints)
        fp.write('Name\n')
        for p in points:
            fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))


def read_xyz(
        filename: str
) -> np.array:
    t = np.loadtxt(
        filename,
        skiprows=2,
        usecols=(1, 2, 3),
        delimiter=" "
    )
    return t


# # Removed for now because mmcif does not exist for Windows
# def read_mmcif(
#         filename: str
# ):
#     data = []
#     with chisurf.fio.zipped.open_maybe_zipped(
#             filename=filename,
#             mode='r'
#     ) as fp:
#         reader = mmcif.fio.PdbxReader.PdbxReader(fp)
#         reader.read(data)
#     mmcif_atoms = data[0]['atom_site']
#     n_atoms = len(mmcif_atoms)
#     atoms = np.zeros(n_atoms, dtype={'names': keys, 'formats': formats})
#     for i, d in enumerate(mmcif_atoms):
#         if d[0] == "ATOM":
#             atoms[i]['xyz'] = float(d[10]), float(d[11]), float(d[12])
#             atoms[i]['i'] = int(d[1])
#             atoms[i]['element'] = int(d[2])
#         else:
#             continue
#
