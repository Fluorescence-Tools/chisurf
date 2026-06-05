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

from chisurf.core.fio.fluorescence import BeckerHicklSetReader


class TestBeckerHicklSetReader(unittest.TestCase):
    """Test case for the BeckerHicklSetReader class."""

    def setUp(self):
        """Use the real bh.set file for testing."""
        # Use the real test data file
        self.set_file = Path(os.path.dirname(__file__)) / "data" / "tttr" / "BH" / "830" / "bh.set"
        
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
        
        # Test micro_time_resolution (now in nanoseconds)
        self.assertAlmostEqual(reader.micro_time_resolution, 0.018313279999999998)
        
        # Test tac_range
        self.assertAlmostEqual(reader.tac_range, 1.5002239e-07)


if __name__ == "__main__":
    unittest.main()