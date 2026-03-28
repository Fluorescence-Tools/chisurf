import utils
import os
import unittest
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *

@unittest.skipUnless(MONGODB_AVAILABLE, "MongoDB not available")
class TestMongoObjectValue(unittest.TestCase):
    
    def test_mongo_object_value_in_json(self):
        """Test that MongoObject JSON includes the value field if it exists in the document"""
        # Create a MongoObject
        obj = cn.MongoObject("test_name")
        
        # Connect to the database
        obj.connect_to_db(**DB_DICT)
        
        # Add a value field to the document
        obj.set_array_int("value", [1, 2, 3, 4, 5])
        
        # Get the JSON representation
        json_str = obj.get_json()
        d = json.loads(json_str)
        
        # Check that the value field is present
        self.assertTrue('value' in d, "JSON should contain value field")
        
        # Check that the value is correct
        self.assertEqual(d['value'], [1, 2, 3, 4, 5], "Value should be [1, 2, 3, 4, 5]")
        
        # Disconnect from the database
        obj.disconnect_from_db()
        
    def test_mongo_object_no_value_in_json(self):
        """Test that MongoObject JSON doesn't include the value field if it doesn't exist in the document"""
        # Create a MongoObject
        obj = cn.MongoObject("test_name")
        
        # Connect to the database
        obj.connect_to_db(**DB_DICT)
        
        # Get the JSON representation
        json_str = obj.get_json()
        d = json.loads(json_str)
        
        # Check that the value field is not present
        self.assertFalse('value' in d, "JSON should not contain value field")
        
        # Disconnect from the database
        obj.disconnect_from_db()
        
if __name__ == '__main__':
    unittest.main()