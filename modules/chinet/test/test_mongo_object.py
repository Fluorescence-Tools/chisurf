import utils
import os
import unittest
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *



class Tests(unittest.TestCase):

    def test_mongo_init(self):
        mo_name = "test_name"
        mo = cn.MongoObject()
        mo.name = mo_name
        self.assertEqual(
            mo.name,
            cn.MongoObject(mo_name).name
        )

    @unittest.skipUnless(CONNECTS, "Cloud not connect to DB")
    def test_mongo_db_connect(self):
        mo = cn.MongoObject()
        self.assertEqual(mo.connect_to_db(**DB_DICT), True)
        self.assertEqual(mo.is_connected_to_db, True)
        mo.write_to_db()
        mo.disconnect_from_db()
        self.assertEqual(mo.is_connected_to_db, False)

        mo2 = cn.MongoObject()
        mo.connect_object_to_db_mongo(mo2)
        self.assertEqual(mo2.is_connected_to_db, True)
        mo2.disconnect_from_db()
        self.assertEqual(mo2.is_connected_to_db, False)

        # mo3 = cn.MongoObject()
        # mo3.connect_to_db(
        #     **db_dict
        # )
        # mo3.read_from_db(mo.get_oid())
        # self.assertEqual(mo3.read_from_db(mo.get_oid()), True)

    def test_mongo_oid(self):
        mo = cn.MongoObject()
        self.assertEqual(len(mo.oid), 24)

    def test_mongo_json(self):
        mo = cn.MongoObject("test_name")
        d = json.loads(mo.get_json())
        self.assertEqual(set(d.keys()), set(['_id', 'precursor', 'death', 'name']))

    def test_singleton(self):
        mo = cn.MongoObject()
        mo.set_singleton_double("d", 22.3)
        self.assertAlmostEqual(mo.get_singleton_double("d"), 22.3)

        mo.set_singleton_int("i", 13)
        self.assertEqual(mo.get_singleton_int("i"), 13)

        mo.set_singleton_bool("b1", True)
        self.assertEqual(mo.get_singleton_bool("b1"), True)

        mo.set_singleton_bool("b2", False)
        self.assertEqual(mo.get_singleton_bool("b2"), False)

    def test_array(self):
        mo = cn.MongoObject()
        mo.set_array_double("d", (1.1, 2.2))
        self.assertTupleEqual(mo.get_array_double("d"), (1.1, 2.2))
        mo.set_array_int("i", [3, 4])
        self.assertEqual(mo.get_array_int("i"), (3, 4))

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
