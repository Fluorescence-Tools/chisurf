import unittest

import utils

import sys
TOPDIR = "."
sys.path.insert(0, TOPDIR)

from chisurf.core.project.registry import Registry, get_registry, reset_registry


class TestRegistry(unittest.TestCase):

    def setUp(self):
        reset_registry()

    def tearDown(self):
        reset_registry()

    def test_dataset_registration(self):
        reg = get_registry()
        obj = {"name": "test_dataset"}
        reg.register_dataset("ds-1", obj)
        self.assertEqual(reg.get_dataset("ds-1"), obj)
        self.assertIn("ds-1", reg.list_datasets())

    def test_dataset_unregistration(self):
        reg = get_registry()
        obj = {"name": "test_dataset"}
        reg.register_dataset("ds-1", obj)
        reg.unregister_dataset("ds-1")
        self.assertIsNone(reg.get_dataset("ds-1"))
        self.assertNotIn("ds-1", reg.list_datasets())

    def test_fit_registration(self):
        reg = get_registry()
        obj = {"name": "test_fit"}
        reg.register_fit("fit-1", obj)
        self.assertEqual(reg.get_fit("fit-1"), obj)
        self.assertIn("fit-1", reg.list_fits())

    def test_fit_unregistration(self):
        reg = get_registry()
        obj = {"name": "test_fit"}
        reg.register_fit("fit-1", obj)
        reg.unregister_fit("fit-1")
        self.assertIsNone(reg.get_fit("fit-1"))
        self.assertNotIn("fit-1", reg.list_fits())

    def test_parameter_registration(self):
        reg = get_registry()
        obj = {"name": "tau", "value": 1.5}
        reg.register_parameter("param-1", obj)
        self.assertEqual(reg.get_parameter("param-1"), obj)
        self.assertIn("param-1", reg.list_parameters())

    def test_parameter_unregistration(self):
        reg = get_registry()
        obj = {"name": "tau", "value": 1.5}
        reg.register_parameter("param-1", obj)
        reg.unregister_parameter("param-1")
        self.assertIsNone(reg.get_parameter("param-1"))
        self.assertNotIn("param-1", reg.list_parameters())

    def test_lifecycle_hooks(self):
        reg = get_registry()
        hook_called = []

        def on_created(uid, obj):
            hook_called.append(("created", uid))

        def on_removed(uid, obj):
            hook_called.append(("removed", uid))

        reg.add_hook("on_dataset_created", on_created)
        reg.add_hook("on_dataset_removed", on_removed)

        reg.register_dataset("ds-1", {"name": "test"})
        self.assertEqual(hook_called, [("created", "ds-1")])

        reg.unregister_dataset("ds-1")
        self.assertEqual(hook_called, [("created", "ds-1"), ("removed", "ds-1")])

    def test_get_stats(self):
        reg = get_registry()
        reg.register_dataset("ds-1", {})
        reg.register_fit("fit-1", {})
        reg.register_parameter("p-1", {})

        stats = reg.get_stats()
        self.assertEqual(stats["datasets"], 1)
        self.assertEqual(stats["fits"], 1)
        self.assertEqual(stats["parameters"], 1)
        self.assertEqual(stats["windows"], 0)

    def test_clear(self):
        reg = get_registry()
        reg.register_dataset("ds-1", {})
        reg.register_fit("fit-1", {})
        reg.clear()

        stats = reg.get_stats()
        self.assertEqual(stats["datasets"], 0)
        self.assertEqual(stats["fits"], 0)
        self.assertEqual(stats["parameters"], 0)


if __name__ == "__main__":
    unittest.main()
