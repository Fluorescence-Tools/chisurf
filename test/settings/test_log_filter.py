"""
Test script for the log filtering functionality.
This script generates log messages with different prefixes to test the filter.
"""

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generate test log messages
logging.info("TEST-INFO: This is a test info message")
logging.warning("TEST-WARNING: This is a test warning message")
logging.error("TEST-ERROR: This is a test error message")
logging.info("FILTER-ME: This message should be filtered when searching for 'FILTER'")
logging.info("ANOTHER-INFO: This is another info message")
logging.warning("ANOTHER-WARNING: This is another warning message")

print("Log messages generated. Please test the filter functionality in the GUI:")
print("1. Type 'TEST' in the filter box - should show only TEST messages")
print("2. Type 'FILTER' in the filter box - should show only the FILTER-ME message")
print("3. Type 'WARNING' in the filter box - should show only warning messages")
print("4. Clear the filter box - should show all messages")