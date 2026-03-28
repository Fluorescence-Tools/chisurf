import utils
import os
import unittest
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *

class TestPortJson(unittest.TestCase):
    
    def test_port_json_includes_value(self):
        """Test that Port JSON includes the value key"""
        # Create a Port object
        port = cn.Port()
        
        # Set a value
        port.set_value_vector([1, 2, 3, 4, 5])
        
        # Get the JSON representation
        json_str = port.get_json()
        d = json.loads(json_str)
        
        # Check that the value key is present
        self.assertTrue('value' in d, "JSON should contain value field")
        
        # Check that the value is correct
        self.assertEqual(len(d['value']), 5, "Value should have 5 elements")
        self.assertEqual(d['value'], [1, 2, 3, 4, 5], "Value should be [1, 2, 3, 4, 5]")
        
    def test_port_json_includes_other_fields(self):
        """Test that Port JSON includes other important fields"""
        # Create a Port object with specific settings
        port = cn.Port(fixed=True, is_output=True, is_reactive=True, is_bounded=True, lb=0, ub=10)
        
        # Get the JSON representation
        json_str = port.get_json()
        d = json.loads(json_str)
        
        # Check that important fields are present
        self.assertTrue('fixed' in d, "JSON should contain fixed field")
        self.assertTrue('is_output' in d, "JSON should contain is_output field")
        self.assertTrue('is_reactive' in d, "JSON should contain is_reactive field")
        self.assertTrue('is_bounded' in d, "JSON should contain is_bounded field")
        self.assertTrue('bounds' in d, "JSON should contain bounds field")
        self.assertTrue('value_type' in d, "JSON should contain value_type field")
        
        # Check field values
        self.assertEqual(d['fixed'], True)
        self.assertEqual(d['is_output'], True)
        self.assertEqual(d['is_reactive'], True)
        self.assertEqual(d['is_bounded'], True)
        self.assertEqual(d['bounds'], [0, 10])

if __name__ == '__main__':
    unittest.main()