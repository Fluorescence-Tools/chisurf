#!/usr/bin/env python
"""
Test script to verify that Qt widgets are properly skipped during serialization.
"""

import sys
import logging
from qtpy.QtWidgets import QApplication, QWidget, QSpinBox
from chisurf.base import Base

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class TestWidget(Base):
    """Test class that contains Qt widgets."""
    
    def __init__(self):
        super().__init__(name="TestWidget")
        self.spinbox = QSpinBox()
        self.regular_value = 42
        self.text_value = "This is a test"

def main():
    """Main function to test Qt widget serialization."""
    # Initialize Qt application
    app = QApplication(sys.argv)
    
    # Create test widget
    test_widget = TestWidget()
    
    # Try to serialize without skipping Qt widgets
    logging.info("Attempting to serialize without skipping Qt widgets...")
    try:
        result = test_widget.to_dict(convert_values_to_elementary=True)
        logging.info("Serialization succeeded without skipping Qt widgets")
    except Exception as e:
        logging.error(f"Error during serialization without skipping: {e}")
    
    # Try to serialize with skipping Qt widgets
    logging.info("Attempting to serialize with skipping Qt widgets...")
    try:
        result = test_widget.to_dict(convert_values_to_elementary=True, skip_qt_widgets=True)
        logging.info("Serialization succeeded with skipping Qt widgets")
        logging.info(f"Result: {result}")
    except Exception as e:
        logging.error(f"Error during serialization with skipping: {e}")
    
    # Try to save to YAML with skipping Qt widgets
    logging.info("Attempting to save to YAML with skipping Qt widgets...")
    try:
        test_widget.save("test_widget.yaml", skip_qt_widgets=True)
        logging.info("Saving to YAML succeeded with skipping Qt widgets")
    except Exception as e:
        logging.error(f"Error during saving to YAML with skipping: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())