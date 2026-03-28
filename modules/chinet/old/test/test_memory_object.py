import utils
import os
import unittest
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *

class TestMemoryObject(unittest.TestCase):
    
    def test_memory_object_json_fields(self):
        """Test that MemoryObject JSON includes all required fields"""
        # Create a MemoryObject directly (not through DatabaseObject)
        obj = cn.MemoryObject("test_name")
        
        # Get the JSON representation
        json_str = obj.get_json()
        d = json.loads(json_str)
        
        # Check that all required fields are present
        self.assertTrue('_id' in d, "JSON should contain _id field")
        self.assertTrue('precursor' in d, "JSON should contain precursor field")
        self.assertTrue('death' in d, "JSON should contain death field")
        self.assertTrue('name' in d, "JSON should contain name field")
        
        # Check field values
        self.assertEqual(d['name'], "test_name")
        self.assertEqual(d['_id'], obj.get_own_oid())
        self.assertEqual(d['precursor'], obj.get_own_oid())  # Initially same as _id
        self.assertEqual(d['death'], 0)  # Initially 0
        
    def test_memory_object_copy(self):
        """Test that copying a MemoryObject preserves all fields"""
        # Create and connect a MemoryObject
        obj = cn.MemoryObject("original")
        obj.connect_to_db("memory", "memory", "memory", "memory")
        
        # Create a copy
        copy_oid = obj.create_copy_in_db()
        
        # Create a new object and read the copy
        copy_obj = cn.MemoryObject()
        copy_obj.connect_to_db("memory", "memory", "memory", "memory")
        copy_obj.read_from_db(copy_oid)
        
        # Get the JSON representation of the copy
        json_str = copy_obj.get_json()
        d = json.loads(json_str)
        
        # Check that all required fields are present
        self.assertTrue('_id' in d, "JSON should contain _id field")
        self.assertTrue('precursor' in d, "JSON should contain precursor field")
        self.assertTrue('death' in d, "JSON should contain death field")
        self.assertTrue('name' in d, "JSON should contain name field")
        
        # Check field values
        self.assertEqual(d['name'], "original")
        self.assertEqual(d['_id'], copy_oid)
        self.assertEqual(d['precursor'], obj.get_own_oid())  # Should be the original object's OID
        
if __name__ == '__main__':
    unittest.main()