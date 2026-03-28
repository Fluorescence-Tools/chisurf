import utils
import os
import unittest
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *

class TestMemoryObjectSetDocument(unittest.TestCase):
    
    def test_set_document_basic(self):
        """Test that set_document correctly updates the document and object fields"""
        # Create a MemoryObject
        obj = cn.MemoryObject("original_name")
        
        # Create a JSON document with various fields
        doc = {
            "_id": "test_id",
            "precursor": "test_precursor",
            "death": 12345,
            "name": "test_name",
            "custom_field": "custom_value"
        }
        
        # Call set_document with the JSON document
        obj.set_document(doc)
        
        # Get the JSON representation
        json_str = obj.get_json()
        d = json.loads(json_str)
        
        # Verify that the document fields are updated correctly
        self.assertEqual(d["_id"], "test_id")
        self.assertEqual(d["precursor"], "test_precursor")
        self.assertEqual(d["death"], 12345)
        self.assertEqual(d["name"], "test_name")
        self.assertEqual(d["custom_field"], "custom_value")
        
        # Verify that the object fields are updated correctly
        self.assertEqual(obj.get_own_oid(), "test_id")
        self.assertEqual(obj.get_name(), "test_name")
    
    def test_set_document_with_value(self):
        """Test that set_document correctly handles the value field for Port objects"""
        # Create a Port object
        port = cn.Port()
        
        # Set an initial value
        port.set_value_vector([1, 2, 3, 4, 5])
        
        # Create a JSON document with a value field
        doc = {
            "_id": "test_port_id",
            "precursor": "test_port_precursor",
            "death": 12345,
            "name": "test_port",
            "value": [10, 20, 30],
            "value_type": 0,
            "fixed": True,
            "is_output": True,
            "is_reactive": True,
            "is_bounded": True,
            "bounds": [0, 10]
        }
        
        # Call set_document with the JSON document
        port.set_document(doc)
        
        # Get the JSON representation
        json_str = port.get_json()
        d = json.loads(json_str)
        
        # Verify that the document fields are updated correctly
        self.assertEqual(d["_id"], "test_port_id")
        self.assertEqual(d["precursor"], "test_port_precursor")
        self.assertEqual(d["death"], 12345)
        self.assertEqual(d["name"], "test_port")
        self.assertEqual(d["fixed"], True)
        self.assertEqual(d["is_output"], True)
        self.assertEqual(d["is_reactive"], True)
        self.assertEqual(d["is_bounded"], True)
        self.assertEqual(d["bounds"], [0, 10])
        
        # The value field might be updated by Port::get_json, so we don't check it directly
        # Instead, we check that the Port object's value is updated correctly
        values = port.get_value_vector()
        self.assertEqual(list(values), [10, 20, 30])

if __name__ == '__main__':
    unittest.main()