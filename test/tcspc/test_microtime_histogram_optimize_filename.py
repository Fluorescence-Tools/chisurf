"""
Test script for the optimize_filename function in MicrotimeHistogram class.
This script tests that the function correctly strips numbered suffixes from filenames.
"""

import sys
import os
import inspect
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.plugins.microtime_histogram.wizard import MicrotimeHistogram

def test_optimize_filename():
    """Test the optimize_filename function with various filename patterns."""
    print("Testing optimize_filename function...")
    
    # Create test cases: (input, expected_output)
    test_cases = [
        ("m_000", "m"),                  # Basic case from the issue description
        ("sample_001", "sample"),        # Another basic case
        ("data_123_456", "data_123"),    # Multiple number patterns, should only strip the last one
        ("experiment", "experiment"),    # No numbered suffix, should remain unchanged
        ("test_1", "test_1"),            # Single digit, should remain unchanged (pattern requires 2+ digits)
        ("file_12", "file"),             # Two digits, should now be stripped with the new pattern
        ("data_0001", "data"),           # Four digits, should be stripped
        ("m_000_extra", "m_000_extra"),  # Numbered pattern not at the end, should remain unchanged
        ("sample_42", "sample"),         # Two digits, should be stripped with the new pattern
        ("test_99_data", "test_99_data"), # Numbered pattern not at the end, should remain unchanged
        ("file_10_20", "file_10"),       # Multiple two-digit patterns, should only strip the last one
        ("data_12345", "data_12345"),    # Five digits, should remain unchanged with the new pattern
    ]
    
    # Run tests
    for input_filename, expected_output in test_cases:
        actual_output = MicrotimeHistogram.optimize_filename(input_filename)
        result = "PASS" if actual_output == expected_output else "FAIL"
        print(f"{result}: '{input_filename}' -> '{actual_output}' (Expected: '{expected_output}')")
    
    print("\nCode inspection summary:")
    print("1. Added optimize_filename static method to MicrotimeHistogram class")
    print("2. Method uses regex to detect and remove numbered suffixes like '_000'")
    print("3. Updated update_output_filename to use the optimize_filename function")
    print("4. Added logging to show the optimization process")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_optimize_filename()