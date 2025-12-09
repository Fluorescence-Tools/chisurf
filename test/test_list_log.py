"""
Test script for the new QListWidget-based log display.
This script generates log messages to test line selection behavior.
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

print("Log messages generated. Please test the line selection behavior:")
print("1. Click on a log line - it should select the entire line")
print("2. Shift+click on another line - it should select a range of lines")
print("3. Ctrl+click on multiple lines - it should select multiple individual lines")
print("4. Type 'TEST' in the filter box - should show only TEST messages")
print("5. Clear the filter box - should show all messages again")