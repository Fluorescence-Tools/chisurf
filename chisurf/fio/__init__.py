"""
The :py:mod:`chisurf.fio` module contains all classes, functions and modules relevant for file input and outputs.
In particular three kinds of file-types are handled:

1. Comma-separated files :py:mod:`chisurf.fio.ascii`
2. PDB-file :py:mod:`chisurf.fio.pdb`
3. TTTR-files containing photon data :py:mod:`chisurf.fio.photons`
4. XYZ-files containing coordinates :py:mod:`chisurf.fio.xyz`
5. DX-files containing densities :py:mod:`chisurf.fio.dx`
6. SDT-files containing time-resolved fluorescence decays :py:mod:`chisurf.fio.sdtfile`

"""
import lzma
import numpy as np

from . zipped import *

import chisurf.fio.fluorescence


def compress_numpy_array(array):
    """
    Compresses a NumPy array and returns a dictionary containing the compressed data,
    shape, and data type information.

    Parameters:
    - array: NumPy array to be compressed.

    Returns:
    - Dictionary containing compressed array, shape, and dtype.
    """
    # Convert the array to bytes
    array_bytes = array.tobytes()

    # Compress the bytes using lzma
    compressed_bytes = lzma.compress(array_bytes)

    # Convert compressed bytes to a base64-encoded string for JSON
    compressed_string = compressed_bytes.hex()

    # Create a dictionary to store the compressed data
    compressed_data = {
        'compressed_array': compressed_string,
        'shape': array.shape,
        'dtype': str(array.dtype)
    }

    return compressed_data

def decompress_numpy_array(compressed_data):
    """
    Decompresses a NumPy array from the compressed data dictionary.

    Parameters:
    - compressed_data: Dictionary containing compressed array, shape, and dtype.

    Returns:
    - Reconstructed NumPy array.
    """
    # Convert the base64-encoded string back to bytes
    compressed_string = compressed_data['compressed_array']
    compressed_bytes = bytes.fromhex(compressed_string)

    # Decompress the bytes using lzma
    decompressed_bytes = lzma.decompress(compressed_bytes)

    # Convert the decompressed bytes back to a NumPy array
    shape = compressed_data['shape']
    dtype = np.dtype(compressed_data['dtype'])
    reconstructed_array = np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)

    return reconstructed_array

