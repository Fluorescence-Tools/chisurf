#!/usr/bin/env python
"""
Test script to verify that Qt widgets are properly skipped during serialization.
"""

import sys
import logging
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtWidgets import QApplication, QWidget, QSpinBox
import chisurf.base

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create a simple class that contains Qt widgets
class TestClass(chisurf.base.Base):
    def __init__(self, name="TestClass"):
        super().__init__(name=name)
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.widget = QWidget()
        self.spinbox = QSpinBox()
        self.normal_attr = "This is a normal attribute"
        self.number = 42

def test_with_skip_qt_widgets():
    """Test serialization with skip_qt_widgets=True"""
    logging.info("Testing serialization with skip_qt_widgets=True")
    test_obj = TestClass()
    
    # Try to convert to dict with skip_qt_widgets=True
    result = test_obj.to_dict(skip_qt_widgets=True)
    
    # Check that Qt widgets were skipped
    assert 'app' not in result, "Qt widget 'app' was not skipped"
    assert 'widget' not in result, "Qt widget 'widget' was not skipped"
    assert 'spinbox' not in result, "Qt widget 'spinbox' was not skipped"
    
    # Check that normal attributes were preserved
    assert 'normal_attr' in result, "Normal attribute was incorrectly skipped"
    assert 'number' in result, "Normal attribute was incorrectly skipped"
    
    logging.info("Test passed: Qt widgets were properly skipped")
    return True

def test_with_to_elementary():
    """Test to_elementary with skip_qt_widgets=True"""
    logging.info("Testing to_elementary with skip_qt_widgets=True")
    test_obj = TestClass()
    
    # Convert to dict first
    d = test_obj.to_dict()
    
    # Then use to_elementary with skip_qt_widgets=True
    result = chisurf.base.to_elementary(d, skip_qt_widgets=True)
    
    # Check that Qt widgets were skipped
    assert 'app' not in result, "Qt widget 'app' was not skipped"
    assert 'widget' not in result, "Qt widget 'widget' was not skipped"
    assert 'spinbox' not in result, "Qt widget 'spinbox' was not skipped"
    
    # Check that normal attributes were preserved
    assert 'normal_attr' in result, "Normal attribute was incorrectly skipped"
    assert 'number' in result, "Normal attribute was incorrectly skipped"
    
    logging.info("Test passed: Qt widgets were properly skipped by to_elementary")
    return True

def test_yaml_serialization():
    """Test YAML serialization with skip_qt_widgets=True"""
    logging.info("Testing YAML serialization with skip_qt_widgets=True")
    test_obj = TestClass()
    
    # Try to convert to YAML with skip_qt_widgets=True
    yaml_str = test_obj.to_yaml(skip_qt_widgets=True)
    
    # Check that the YAML string doesn't contain references to Qt widgets
    assert 'app' not in yaml_str, "Qt widget 'app' was not skipped in YAML"
    assert 'widget' not in yaml_str, "Qt widget 'widget' was not skipped in YAML"
    assert 'spinbox' not in yaml_str, "Qt widget 'spinbox' was not skipped in YAML"
    
    # Check that normal attributes were preserved
    assert 'normal_attr' in yaml_str, "Normal attribute was incorrectly skipped in YAML"
    assert '42' in yaml_str, "Normal attribute was incorrectly skipped in YAML"
    
    logging.info("Test passed: Qt widgets were properly skipped in YAML serialization")
    return True

if __name__ == "__main__":
    try:
        test_with_skip_qt_widgets()
        test_with_to_elementary()
        test_yaml_serialization()
        logging.info("All tests passed!")
    except Exception as e:
        logging.error(f"Test failed: {e}")
        sys.exit(1)
    sys.exit(0)