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

import chisurf.fio as io

import IMP
import IMP.core
import IMP.atom

import chisurf
import chisurf.common



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
    with io.zipped.open_maybe_zipped(
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


def find_atom_index(
        atoms: np.array,
        chain_identifier: str,
        residue_seq_number: int,
        atom_name: str,
        residue_name: str,
        verbose: bool = False,
        ignore_multiple_selections: bool = True
):
    """
    Find the index of an atom by a set of identifiers

    :param atoms:
    :param chain_identifier:
    :param residue_seq_number:
    :param atom_name:
    :param residue_name:
    :param ignore_multiple_selections:
    :param verbose:
    :return:
    """
    # Determine Labeling position
    if residue_seq_number is None or atom_name is None:
        raise ValueError(
            "Either attachment_atom number or residue number and atom_name need to be provided."
        )
    if verbose:
        print("find_atom_index:")
        print("-- Chain ID: %s" % chain_identifier)
        print("-- Residue seq. number: %s" % residue_seq_number)
        print("-- Residue name: %s" % residue_name)
        print("-- Atom name: %s" % atom_name)
    if chain_identifier is None or chain_identifier == '':
        attachment_atom_index = np.where(
            (atoms['res_id'] == residue_seq_number) &
            (atoms['atom_name'] == atom_name)
        )[0]
        if verbose:
            print(
                "-- WARNING no chain specified. Possible attachment atoms: % s"
                % attachment_atom_index
            )
    else:
        attachment_atom_index = np.where(
            (atoms['res_id'] == residue_seq_number) &
            (atoms['atom_name'] == atom_name) &
            (atoms['chain'] == chain_identifier)
        )[0]
    if len(attachment_atom_index) != 1 and not ignore_multiple_selections:
        print("Search atom index:")
        print("-- Chain ID: %s" % chain_identifier)
        print("-- Residue seq. number: %s" % residue_seq_number)
        print("-- Residue name: %s" % residue_name)
        print("-- Atom name: %s" % atom_name)
        raise ValueError("Invalid selection of attachment atom")
    else:
        attachment_atom_index = attachment_atom_index[0]
    if verbose:
        print("Atom index: %s" % attachment_atom_index)
    return attachment_atom_index


get_atom_index = find_atom_index



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



def convert_atoms(
        ps: typing.List[IMP.atom.Hierarchy],
        radius_no_interaction: bool = True
) -> np.ndarray:
    """Converts a list of IMP.atom.Hierarchy to a numpy record array

    Parameters
    ----------
    ps: list of IMP.atom.Hierarchy

    radius_no_interaction: bool
        If set to True the returned radii are the radii where the potential
        energy is zero. Otherwise, the radii are correspond to the minimal
        distance Rmin in a 6-12 LJ potential
        :math:`E = eij ((Rmin/rij)**12 - 2*(Rmin/rij)**6))`

    Returns
    -------
    atoms: numpy array containing the atom information

    """
    atoms = np.zeros(
        len(ps),
        dtype={
            'names': keys,
            'formats': formats
        }
    )
    radius_scaleling = 1.0
    if radius_no_interaction:
        # IMP uses the CHARM force field radius Rmin
        # E = eij ((Rmin/rij)**12 - 2*(Rmin/rij)**6))
        # The LJ radius is the distance where E = 0
        radius_scaleling = 2**(-1./6.)
    t = IMP.atom.get_element_table()
    for i, atom in enumerate(ps):
        a = IMP.atom.Atom(atom)
        r = IMP.atom.Residue(a.get_parent())
        c = IMP.atom.Chain(r.get_parent())
        atoms[i]['i'] = i
        atoms[i]['chain'] = c.get_id()
        atoms[i]['res_id'] = r.get_index()
        atoms[i]['res_name'] = r.get_name()
        atoms[i]['atom_id'] = a.get_input_index()
        atoms[i]['atom_name'] = str(a.get_atom_type())[1:-1]
        atoms[i]['element'] = t.get_name(a.get_element())
        atoms[i]['xyz'] = IMP.core.XYZR(atom).get_coordinates()
        atoms[i]['radius'] = IMP.core.XYZR(atom).get_radius() * radius_scaleling
        atoms[i]['bfactor'] = a.get_temperature_factor()
        atoms[i]['mass'] = IMP.atom.Mass(atom).get_mass()
    return atoms



def read_coordinates(
    filename: str
) -> np.ndarray:
    """

    Examples
    --------
    >>> import chisurf as cs
    >>> import chisurf.fio
    >>> atoms = cs.fio.structure.read_coordinates('./test/data/1fat.cif')

    :param filename:
    :return:
    """
    model = IMP.Model()
    if not os.path.isfile(filename):
        raise FileNotFoundError("The file %s could not be found." % filename)
    if filename.upper().endswith('.PDB'):
        mp = IMP.atom.read_pdb(filename, model, IMP.atom.NonWaterPDBSelector())
        return convert_atoms(
            IMP.atom.get_by_type(mp, IMP.atom.ATOM_TYPE)
        )
    if filename.upper().endswith('.CIF'):
        print("Opening ci")
        mp = IMP.atom.read_mmcif(filename, model, IMP.atom.NonWaterPDBSelector())
        return convert_atoms(
            IMP.atom.get_by_type(mp, IMP.atom.ATOM_TYPE)
        )


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
        with io.zipped.open_maybe_zipped(
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
                atoms = read_coordinates(
                    filename=filename
                )
            return atoms
    else:
        return np.zeros(0, dtype={'names': keys, 'formats': formats})
