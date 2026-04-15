# Consolidated test file: test_bhfiles.py


# --- FROM test_bhfiles.py ---

# --- FROM test_becker_hickl_set.py ---
"""
Test script for the BeckerHicklSetReader class.

This script tests the basic functionality of the BeckerHicklSetReader class
using a real .set file from the test data directory.
"""

import os
import sys
import tempfile
from pathlib import Path
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chisurf.fio.fluorescence import BeckerHicklSetReader


class TestBeckerHicklSetReader(unittest.TestCase):
    """Test case for the BeckerHicklSetReader class."""

    def setUp(self):
        """Use the real bh.set file for testing."""
        # Use the real test data file
        self.set_file = Path(os.path.dirname(__file__)) / ".." / "data" / "tttr" / "BH" / "830" / "bh.set"
        
        # Verify the file exists
        if not self.set_file.exists():
            self.fail(f"Test file not found: {self.set_file}")
    
    
    def test_read_parameters(self):
        """Test that parameters are correctly read from the .set file."""
        reader = BeckerHicklSetReader(self.set_file)
        
        # Test raw parameter access
        self.assertAlmostEqual(reader.get_param('SYN_FQ'), -50.980392)
        self.assertAlmostEqual(reader.get_param('TAC_TC'), 1.831328e-11)
        self.assertAlmostEqual(reader.get_param('TAC_R'), 1.5002239e-07)
        self.assertEqual(reader.get_param('ADC_RE'), 4096)
        
        # Test default value for non-existent parameter
        self.assertIsNone(reader.get_param('NONEXISTENT'))
        self.assertEqual(reader.get_param('NONEXISTENT', 'default'), 'default')
    
    def test_derived_properties(self):
        """Test that derived properties are correctly calculated."""
        reader = BeckerHicklSetReader(self.set_file)
        
        # Test macro_time_resolution
        expected_macro_time = 1.0 / (abs(-50.980392) * 1e6)  # 1 / |SYN_FQ| in MHz
        self.assertAlmostEqual(reader.macro_time_resolution, expected_macro_time)
        
        # Test micro_time_resolution (in seconds)
        self.assertAlmostEqual(reader.micro_time_resolution, 1.831328e-11)
        
        # Test tac_range
        self.assertAlmostEqual(reader.tac_range, 1.5002239e-07)



# --- FROM test_becker_hickl_set_simple.py ---
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

