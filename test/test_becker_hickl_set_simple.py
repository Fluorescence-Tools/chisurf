"""
Simple test script for the BeckerHicklSetReader class.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.fio.fluorescence.becker_hickl_set import BeckerHicklSetReader

def main():
    # Create a temporary .set file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        set_file = Path(temp_dir) / "test.set"
        
        # Create a simple .set file with known parameters
        content = """
        [SP_SYN_FQ,F,-50.98]
        [SP_TAC_TC,F,1.83e-11]
        [SP_TAC_R,F,5.0e-8]
        [SP_ADC_RE,I,4096]
        """
        set_file.write_text(content)
        
        # Test the reader
        reader = BeckerHicklSetReader(set_file)
        
        # Print the parameters
        print("Raw parameters:")
        print(f"  SYN_FQ: {reader.get_param('SYN_FQ')}")
        print(f"  TAC_TC: {reader.get_param('TAC_TC')}")
        print(f"  TAC_R: {reader.get_param('TAC_R')}")
        print(f"  ADC_RE: {reader.get_param('ADC_RE')}")
        
        # Print the derived properties
        print("\nDerived properties:")
        print(f"  Macro time resolution: {reader.macro_time_resolution:.3e} s")
        print(f"  Micro time resolution: {reader.micro_time_resolution:.3e} s")
        print(f"  TAC range: {reader.tac_range:.3e} s")
        
        # Print a summary
        print("\nSummary:")
        reader.summary()
        
        print("\nTest completed successfully!")

if __name__ == "__main__":
    main()