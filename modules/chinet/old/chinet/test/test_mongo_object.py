import utils
import os
import unittest
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *



class Tests(unittest.TestCase):

    def test_database_init(self):
        """Test initialization of DatabaseObject"""
        obj_name = "test_name"
        obj = cn.DatabaseObject()
        obj.name = obj_name
        self.assertEqual(
            obj.name,
            cn.DatabaseObject(obj_name).name
        )

    @unittest.skipUnless(CONNECTS, "Could not connect to DB")
    def test_database_connect(self):
        """Test connecting to database (MongoDB or in-memory)"""
        obj = cn.DatabaseObject()
        if WITH_MONGODB:
            self.assertEqual(obj.connect_to_db(**DB_DICT), True)
        else:
            self.assertEqual(obj.connect_to_db("memory", "memory", "memory", "memory"), True)
        self.assertEqual(obj.is_connected_to_db, True)
        obj.write_to_db()
        obj.disconnect_from_db()
        self.assertEqual(obj.is_connected_to_db, False)

        obj2 = cn.DatabaseObject()
        if WITH_MONGODB:
            obj.connect_object_to_db_mongo(obj2)
        else:
            obj.connect_object_to_db(obj2)
        self.assertEqual(obj2.is_connected_to_db, True)
        obj2.disconnect_from_db()
        self.assertEqual(obj2.is_connected_to_db, False)

    def test_database_oid(self):
        """Test object ID generation"""
        obj = cn.DatabaseObject()
        # OID should be a non-empty string
        self.assertTrue(len(obj.oid) > 0)

    def test_database_json(self):
        """Test JSON serialization"""
        obj = cn.DatabaseObject("test_name")
        d = json.loads(obj.get_json())
        # All objects should have at least _id and name
        self.assertTrue('_id' in d)
        self.assertTrue('name' in d)
        self.assertEqual(d['name'], "test_name")

    def test_singleton(self):
        """Test singleton values (double, int, bool)"""
        obj = cn.DatabaseObject()
        obj.set_singleton_double("d", 22.3)
        self.assertAlmostEqual(obj.get_singleton_double("d"), 22.3)

        obj.set_singleton_int("i", 13)
        self.assertEqual(obj.get_singleton_int("i"), 13)

        obj.set_singleton_bool("b1", True)
        self.assertEqual(obj.get_singleton_bool("b1"), True)

        obj.set_singleton_bool("b2", False)
        self.assertEqual(obj.get_singleton_bool("b2"), False)

    def test_array(self):
        """Test array values (double, int)"""
        obj = cn.DatabaseObject()
        obj.set_array_double("d", (1.1, 2.2))
        self.assertTupleEqual(obj.get_array_double("d"), (1.1, 2.2))
        obj.set_array_int("i", [3, 4])
        self.assertEqual(obj.get_array_int("i"), (3, 4))

    # TODO: NOT READY
    # @unittest.expectedFailure
    # def test_read_json(self):
    #     json_file = "inputs/session_template.json"

    #     json_string = ""
    #     with open(json_file, 'r') as fp:
    #         json_string = fp.read()

    #     # contains node & links dict
    #     mo1 = cn.MongoObject()
    #     mo1.read_json(json_string)

    #     # contains only node dict
    #     mo2 = cn.MongoObject()
    #     mo2.read_json(mo1.get_json())
    #     sub_json = mo1["nodes"].get_json()

    #     superset = json.loads(mo1.get_json())
    #     subset = json.loads(mo2.get_json())

    #     self.assertEqual(
    #         all(item in superset.items() for item in subset.items()),
    #         True
    #     )

    #     subset = json.loads(json_string)
    #     superset = json.loads(mo["nodes"].get_json())
    #     self.assertEqual(
    #         all(item in superset.items() for item in subset.items()),
    #         True
    #     )


if __name__ == '__main__':
    unittest.main()
