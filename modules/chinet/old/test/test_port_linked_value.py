import utils
import os
import unittest
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *

class TestPortLinkedValue(unittest.TestCase):
    
    def test_port_json_includes_linked_value(self):
        """Test that Port JSON includes the value from a linked Port"""
        # Create two Port objects
        port1 = cn.Port(name="port1")
        port2 = cn.Port(name="port2")
        
        # Set values for both ports
        port1.set_value_vector([1, 2, 3])
        port2.set_value_vector([4, 5, 6])
        
        # Link port1 to port2
        port1.set_link(port2)
        
        # Get the JSON representation of port1
        json_str = port1.get_json()
        d = json.loads(json_str)
        
        # Check that the value key is present
        self.assertTrue('value' in d, "JSON should contain value field")
        
        # Check that the value is from port2 (the linked port)
        self.assertEqual(d['value'], [4, 5, 6], "Value should be from the linked port")
        
    def test_port_json_includes_own_value_when_not_linked(self):
        """Test that Port JSON includes its own value when not linked"""
        # Create a Port object
        port = cn.Port()
        
        # Set a value
        port.set_value_vector([1, 2, 3])
        
        # Get the JSON representation
        json_str = port.get_json()
        d = json.loads(json_str)
        
        # Check that the value key is present
        self.assertTrue('value' in d, "JSON should contain value field")
        
        # Check that the value is correct
        self.assertEqual(d['value'], [1, 2, 3], "Value should be the port's own value")
        
    def test_port_json_includes_linked_value_after_update(self):
        """Test that Port JSON includes the updated value from a linked Port"""
        # Create two Port objects
        port1 = cn.Port(name="port1")
        port2 = cn.Port(name="port2")
        
        # Set initial values
        port1.set_value_vector([1, 2, 3])
        port2.set_value_vector([4, 5, 6])
        
        # Link port1 to port2
        port1.set_link(port2)
        
        # Update port2's value
        port2.set_value_vector([7, 8, 9])
        
        # Get the JSON representation of port1
        json_str = port1.get_json()
        d = json.loads(json_str)
        
        # Check that the value key is present
        self.assertTrue('value' in d, "JSON should contain value field")
        
        # Check that the value is the updated value from port2
        self.assertEqual(d['value'], [7, 8, 9], "Value should be the updated value from the linked port")

if __name__ == '__main__':
    unittest.main()