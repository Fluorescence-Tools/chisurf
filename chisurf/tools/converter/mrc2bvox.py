#!/usr/bin/env python
import mrcfile
import numpy as np
import glob
import argparse
import os


parser = argparse.ArgumentParser(
    description='Convert a MRC file to a bvox file for Blender.'
)
parser.add_argument('--input_files', type=str, help='a set of PDB files')
args = parser.parse_args()
opt = vars(args)

input_files = glob.glob(args.input_files)

for input_file in input_files:
    mrc = mrcfile.mfm.io.zipped.open_maybe_zipped(input_file)
    nx, ny, nz = mrc.data.shape
    header = np.array([nx, ny, nz, 1])
    output_file = "".join(os.path.abspath(input_file).split(".")[:-1]) + ".bvox"
    with open(output_file, 'wb') as binfile:
        header.astype('<i4').tofile(binfile)
        mrc.data.astype('<f4').tofile(binfile)
