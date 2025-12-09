"""
Test script to verify that BeckerHicklSetReader.micro_time_resolution returns values in nanoseconds.

This script creates a temporary .set file with known parameters and verifies that
the micro_time_resolution property returns the expected value in nanoseconds.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.fio.fluorescence import BeckerHicklSetReader

def main():
    """Test that micro_time_resolution returns values in nanoseconds."""
    print("Testing BeckerHicklSetReader.micro_time_resolution...")
    
    # Create a temporary .set file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        set_file = Path(temp_dir) / "test.set"
        
        # Create a simple .set file with known parameters
        # TAC_TC is 1.83e-11 seconds = 18.3 picoseconds = 0.0183 nanoseconds
        content = """
        [SP_SYN_FQ,F,-50.98]
        [SP_TAC_TC,F,1.83e-11]
        [SP_TAC_R,F,5.0e-8]
        [SP_ADC_RE,I,4096]
        """
        set_file.write_text(content)
        
        # Test the reader
        reader = BeckerHicklSetReader(set_file)
        
        # Get the micro_time_resolution
        micro_time_res = reader.micro_time_resolution
        
        # Expected value in nanoseconds: 1.83e-11 seconds * 1e9 = 18.3 nanoseconds
        expected_ns = 1.83e-11 * 1e9
        
        # Print the results
        print(f"Raw TAC_TC parameter: {reader.get_param('TAC_TC'):.3e} seconds")
        print(f"micro_time_resolution: {micro_time_res:.3f} nanoseconds")
        print(f"Expected value: {expected_ns:.3f} nanoseconds")
        
        # Verify the result
        if abs(micro_time_res - expected_ns) < 1e-6:
            print("\nTEST PASSED: micro_time_resolution returns the correct value in nanoseconds")
        else:
            print("\nTEST FAILED: micro_time_resolution does not return the expected value")
            print(f"  Expected: {expected_ns:.3f} ns")
            print(f"  Actual: {micro_time_res:.3f} ns")
        
if __name__ == "__main__":
    main()