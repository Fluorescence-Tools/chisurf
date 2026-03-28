import utils
import os
import unittest
import json
import numpy as np

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import chinet as cn
from constants import *

class Tests(unittest.TestCase):

    def test_port_init_singelton(self):
        v1 = 23.0
        v2 = 29.0
        p1 = cn.Port(v1)
        # check setting of value
        p2 = cn.Port()
        p2.value = v1
        # check fixing
        p3 = cn.Port(
            value=v1,
            fixed=True
        )
        p4 = cn.Port(
            value=v1,
            fixed=False
        )
        # check linking
        p5 = cn.Port(v2)
        p5.link = p4

        self.assertEqual(
            np.allclose(
                p1.value, p2.value
            ),
            True
        )
        self.assertEqual(p3.fixed, True)
        self.assertEqual(p4.fixed, False)
        self.assertEqual(
            np.allclose(
                p5.value,
                p4.value
            ),
            True
        )

    def test_port_init_singelton(self):
        """Test chinet Port class set_value and get_value"""
        v1 = 23.0
        v2 = 29.0
        p1 = cn.Port()
        p1.value = v1
        # check setting of value
        p2 = cn.Port(v1)
        # check fixing
        p3 = cn.Port(
            value=v1,
            fixed=True
        )
        p4 = cn.Port(
            value=v1,
            fixed=False
        )
        # check linking
        p5 = cn.Port(v2)
        p5.link = p4

        self.assertEqual(
            np.allclose(
                p1.value, p2.value
            ),
            True
        )
        self.assertEqual(p3.fixed, True)
        self.assertEqual(p4.fixed, False)
        self.assertEqual(
            np.allclose(
                p5.value,
                p4.value
            ),
            True
        )

        # check bounds
        fixed = False
        is_output = False
        is_reactive = False
        is_bounded = True
        lower_bound = 2
        upper_bound = 5
        value = 0
        p6 = cn.Port(
            value=value,
            fixed=fixed,
            is_output=is_output,
            is_reactive=is_reactive,
            is_bounded=is_bounded,
            lb=lower_bound,
            ub=upper_bound
        )
        self.assertEqual(
            np.all(p6.value <= upper_bound),
            True
        )
        self.assertEqual(
            np.all(p6.value >= lower_bound),
            True
        )
        self.assertAlmostEqual(
            p6.value,
            lower_bound  # the lower bound is not part of the
        )

    def test_port_bounds(self):
        """Test chinet Port class set_value and get_value"""
        v1 = np.array(
            [1, 2, 3, 6, 5.5, -3, -2, -6.1, -10000, 10000],
            dtype=np.double
        )
        p1 = cn.Port()
        p1.value = v1

        self.assertEqual(
            np.allclose(
                p1.value,
                v1
            ),
            True
        )

        p1.bounds = 0, 1
        p1.bounded = True

        self.assertEqual(
            (p1.value <= 1).all(),
            True
        )
        self.assertEqual(
            (p1.value >= 0).all(),
            True
        )

    def test_port_get_set_value(self):
        """Test chinet Port class set_value and get_value"""
        v1 = [1, 2, 3]
        p1 = cn.Port()
        p1.value = v1

        p2 = cn.Port()
        p2.value = v1
        self.assertEqual(
            (p1.value == p2.value).all(),
            True
        )

    def test_port_init_vector(self):
        """Test chinet Port class set_value and get_value"""
        v1 = [1, 2, 3, 5, 8]
        v2 = [1, 2, 4, 8, 16]
        # check setting of value
        p1 = cn.Port()
        p1.value = v1
        p2 = cn.Port(v1)
        self.assertListEqual(
            list(p2.value),
            list(p1.value)
        )
        # check fixing
        p3 = cn.Port(v1, True)
        p4 = cn.Port(v1, False)
        self.assertEqual(p3.fixed, True)
        self.assertEqual(p4.fixed, False)

        # check linking
        p5 = cn.Port(v2, False)
        p5.link = p4
        self.assertEqual(
            np.allclose(
                p5.value,
                p4.value
            ),
            True
        )

    def test_port_init_array(self):
        """Test chinet Port class set_value and get_value"""
        array = np.array([1, 2, 3, 5, 8, 13], dtype=np.double)
        p1 = cn.Port()
        p1.value = array
        p2 = cn.Port(array)
        self.assertListEqual(
            list(p1.value),
            list(p2.value)
        )

    def test_set_get_value_1(self):
        """Test chinet Port class set_value and get_value"""
        value = 23.0
        port = cn.Port(value)
        self.assertEqual(port.value, value)

    def test_set_get_value_2(self):
        """Test chinet Port class set_value and get_value"""
        value = (1,)
        port = cn.Port()
        port.value = value
        self.assertEqual(port.value, value)

    def test_port_link_value(self):
        value1 = np.array([12], dtype=np.double)
        value2 = np.array([6], dtype=np.double)
        p1 = cn.Port(value1)
        p2 = cn.Port(value2)

        self.assertEqual(
            np.allclose(p1.value, value1),
            True
        )
        self.assertEqual(
            np.allclose(p2.value, value2),
            True
        )
        self.assertEqual(
            np.allclose(p1.value, p2.value),
            False
        )

        p2.link = p1
        self.assertEqual(
            np.allclose(p1.value, p2.value),
            True
        )
        p2.unlink()
        self.assertEqual(
            np.allclose(p2.value, value2),
            True
        )

    def test_port_fixed(self):
        p1 = cn.Port(12)
        p1.fixed = True
        self.assertEqual(p1.fixed, True)

        p1.fixed = False
        self.assertEqual(p1.fixed, False)

    def test_port_reactive(self):
        p1 = cn.Port(12)
        p1.reactive = True
        self.assertEqual(p1.reactive, True)

        p1.reactive = False
        self.assertEqual(p1.reactive, False)

    @unittest.skipUnless(CONNECTS, "Could not connect to DB")
    def test_db_write(self):
        """Test writing a Port to the database"""
        value_array = (1, 2, 3, 5, 8, 13)
        port = cn.Port(
            value=value_array,
            fixed=True
        )

        # Connect to the appropriate database
        if WITH_MONGODB:
            connect_success = port.connect_to_db(**DB_DICT)
        else:
            connect_success = port.connect_to_db("memory", "memory", "memory", "memory")

        write_success = port.write_to_db()

        self.assertEqual(connect_success, True)
        self.assertEqual(write_success, True)

    @unittest.skipUnless(CONNECTS, "Could not connect to DB")
    def test_port_db_restore(self):
        """Test reading a Port from the database"""
        value_array = (1, 2, 3, 5, 8, 13)
        value = 17

        port = cn.Port()
        port.value = value
        port.value = value_array

        # Connect to the appropriate database
        if WITH_MONGODB:
            port.connect_to_db(**DB_DICT)
        else:
            port.connect_to_db("memory", "memory", "memory", "memory")

        port.write_to_db()

        port_reload = cn.Port()
        if WITH_MONGODB:
            port_reload.connect_to_db(**DB_DICT)
        else:
            port_reload.connect_to_db("memory", "memory", "memory", "memory")

        self.assertEqual(port_reload.read_from_db(port.oid), True)

        dict_port = json.loads(port.get_json())
        dict_port_restore = json.loads(port_reload.get_json())

        self.assertEqual(dict_port, dict_port_restore)


    def test_port_type_conversion_auto(self):
        """Test automatic type conversion when assigning values"""
        # Create an int port
        p1 = cn.Port(value=42)
        self.assertEqual(p1.get_value_type(), 0)  # 0 = int type

        # Assign a float value, should automatically upcast to float
        p1.value = 42.5
        self.assertEqual(p1.get_value_type(), 1)  # 1 = float type
        self.assertEqual(p1.value, 42.5)

        # Create a float port
        p2 = cn.Port(value=42.5)
        self.assertEqual(p2.get_value_type(), 1)  # 1 = float type

        # Assign an int value, should keep float type (no downcast)
        p2.value = 42
        self.assertEqual(p2.get_value_type(), 1)  # Still float type
        self.assertEqual(p2.value, 42.0)  # Value should be converted to float

    def test_port_type_conversion_explicit(self):
        """Test explicit type conversion using dtype property"""
        # Create a port with default type
        p1 = cn.Port(value=42)
        self.assertEqual(p1.get_value_type(), 0)  # 0 = int type

        # Explicitly change type to float
        p1.dtype = np.float64
        self.assertEqual(p1.get_value_type(), 1)  # 1 = float type
        self.assertEqual(p1.value, 42.0)  # Value should be converted to float

        # Create a float port
        p2 = cn.Port(value=42.5)
        self.assertEqual(p2.get_value_type(), 1)  # 1 = float type

        # Explicitly change type to int (downcast)
        p2.dtype = np.int64
        self.assertEqual(p2.get_value_type(), 0)  # 0 = int type
        self.assertEqual(p2.value, 42)  # Value should be truncated to int

    def test_port_type_conversion_array(self):
        """Test type conversion with array values"""
        # Create an int port with array values
        p1 = cn.Port(value=np.array([1, 2, 3, 4, 5]))
        self.assertEqual(p1.get_value_type(), 0)  # 0 = int type

        # Assign a float array, should automatically upcast to float
        p1.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        self.assertEqual(p1.get_value_type(), 1)  # 1 = float type
        self.assertTrue(np.allclose(p1.value, np.array([1.1, 2.2, 3.3, 4.4, 5.5])))

        # Create a float port with array values
        p2 = cn.Port(value=np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
        self.assertEqual(p2.get_value_type(), 1)  # 1 = float type

        # Assign an int array, should keep float type (no downcast)
        p2.value = np.array([1, 2, 3, 4, 5])
        self.assertEqual(p2.get_value_type(), 1)  # Still float type
        self.assertTrue(np.allclose(p2.value, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))

    def test_port_type_conversion_memory(self):
        """Test memory allocation during type conversion"""
        # Create a large int array
        large_array = np.arange(1000, dtype=np.int64)
        p1 = cn.Port(value=large_array)
        self.assertEqual(p1.get_value_type(), 0)  # 0 = int type

        # Get the current buffer size
        initial_size = p1.current_size()

        # Convert to float (should allocate more memory)
        p1.dtype = np.float64
        self.assertEqual(p1.get_value_type(), 1)  # 1 = float type

        # Verify all values were preserved
        self.assertTrue(np.allclose(p1.value, large_array.astype(np.float64)))

        # Create a large float array
        large_float_array = np.arange(1000, dtype=np.float64) + 0.5
        p2 = cn.Port(value=large_float_array)
        self.assertEqual(p2.get_value_type(), 1)  # 1 = float type

        # Convert to int (should truncate values)
        p2.dtype = np.int64
        self.assertEqual(p2.get_value_type(), 0)  # 0 = int type

        # Verify values were truncated
        expected = large_float_array.astype(np.int64)
        self.assertTrue(np.allclose(p2.value, expected))

    def test_port_type_conversion_edge_cases(self):
        """Test type conversion edge cases"""
        # Empty port
        p1 = cn.Port()
        p1.dtype = np.float64
        self.assertEqual(p1.get_value_type(), 1)  # 1 = float type

        # Very large values
        p2 = cn.Port(value=np.iinfo(np.int64).max)
        self.assertEqual(p2.get_value_type(), 0)  # 0 = int type
        p2.dtype = np.float64
        self.assertEqual(p2.get_value_type(), 1)  # 1 = float type
        self.assertEqual(p2.value, float(np.iinfo(np.int64).max))

        # Very small values
        p3 = cn.Port(value=np.finfo(np.float64).tiny)
        self.assertEqual(p3.get_value_type(), 1)  # 1 = float type
        p3.dtype = np.int64
        self.assertEqual(p3.get_value_type(), 0)  # 0 = int type
        self.assertEqual(p3.value, 0)  # Should be truncated to 0

        # Mixed type arrays
        mixed_array = np.array([1, 2.5, 3, 4.5, 5])
        p4 = cn.Port(value=mixed_array)
        self.assertEqual(p4.get_value_type(), 1)  # Should be float type
        self.assertTrue(np.allclose(p4.value, mixed_array))

    def test_port_type_preservation(self):
        """Test that the dtype setter preserves the port's internal type values (2-ness or 3-ness)"""
        # Create a port with value_type 0 (int)
        p1 = cn.Port(value=42)
        self.assertEqual(p1.get_value_type(), 0)  # 0 = int type

        # Manually set value_type to 2 (another int type)
        p1.set_value_type(2)
        self.assertEqual(p1.get_value_type(), 2)

        # Change dtype to float, should preserve the "2-ness" and become value_type 3
        p1.dtype = np.float64
        self.assertEqual(p1.get_value_type(), 3)  # Should be 3 (float type with "2-ness" preserved)

        # Change back to int, should preserve the "3-ness" and become value_type 2
        p1.dtype = np.int64
        self.assertEqual(p1.get_value_type(), 2)  # Should be 2 (int type with "3-ness" preserved)

        # Create a port with value_type 1 (float)
        p2 = cn.Port(value=42.5)
        self.assertEqual(p2.get_value_type(), 1)  # 1 = float type

        # Manually set value_type to 3 (another float type)
        p2.set_value_type(3)
        self.assertEqual(p2.get_value_type(), 3)

        # Change dtype to int, should preserve the "3-ness" and become value_type 2
        p2.dtype = np.int64
        self.assertEqual(p2.get_value_type(), 2)  # Should be 2 (int type with "3-ness" preserved)

        # Change back to float, should preserve the "2-ness" and become value_type 3
        p2.dtype = np.float64
        self.assertEqual(p2.get_value_type(), 3)  # Should be 3 (float type with "2-ness" preserved)

if __name__ == '__main__':
    unittest.main()
