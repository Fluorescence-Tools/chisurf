from __future__ import annotations

import os
import pathlib

import chisurf.core.fio.fluorescence.pqres
import chisurf.core.experiments.fcs
import chisurf.core.experiments.tcspc

# Test reading FCS pqres file
def test_read_fcs_pqres():
    # Path to the example FCS pqres file
    fcs_file = os.path.join('test', 'data', 'fcs', 'Al488_10uM-KI_150b_2_OFCS.pqres')
    
    # Create an FCS reader
    fcs_reader = chisurf.core.experiments.fcs.FCS()
    
    # Read the file
    data_group = fcs_reader.read(filename=fcs_file, reader_name='pqres')
    
    # Print information about the data group
    print(f"FCS pqres file: {fcs_file}")
    print(f"Number of curves: {len(data_group)}")
    for i, curve in enumerate(data_group):
        print(f"  Curve {i+1}: {curve.name}, {len(curve.x)} points")
    
    return data_group

# Test reading TCSPC pqres file
def test_read_tcspc_pqres():
    # Path to the example TCSPC pqres file
    tcspc_file = os.path.join('test', 'data', 'tcspc', 'Al488_10uM-KI_150b_2_OTCSPC.pqres')
    
    # Create a TCSPC reader
    tcspc_reader = chisurf.core.experiments.tcspc.TCSPCReader()
    
    # Read the file
    data_group = tcspc_reader.read(filename=tcspc_file)
    
    # Print information about the data group
    print(f"TCSPC pqres file: {tcspc_file}")
    print(f"Number of curves: {len(data_group)}")
    for i, curve in enumerate(data_group):
        print(f"  Curve {i+1}: {curve.name}, {len(curve.x)} points")
    
    return data_group

if __name__ == "__main__":
    print("Testing PQResReader...")
    
    # Test reading FCS pqres file
    fcs_data = test_read_fcs_pqres()
    print()
    
    # Test reading TCSPC pqres file
    tcspc_data = test_read_tcspc_pqres()