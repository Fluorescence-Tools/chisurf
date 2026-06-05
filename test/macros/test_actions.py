import pathlib
import unittest
import sys

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

import chisurf
from chisurf.core.actions._infra import ActionSpec, ActionRegistry, ActionDispatcher
from chisurf.core.actions import dispatch

class TestActions(unittest.TestCase):
    def setUp(self):
        # Setup a clean registry and dispatcher for testing
        self.registry = ActionRegistry()
        self.history = []
        
        class MockHistory:
            def record(self, **kwargs):
                self.history.append(kwargs)
                return kwargs
            
        self.mock_history = MockHistory()
        self.mock_history.history = self.history
        
        self.dispatcher = ActionDispatcher(
            registry=self.registry,
            history_provider=lambda: self.mock_history
        )
        
        # Patch chisurf to use our test dispatcher
        self.old_dispatcher = getattr(chisurf, "action_dispatcher", None)
        chisurf.action_dispatcher = self.dispatcher
        chisurf.action_registry = self.registry

    def tearDown(self):
        chisurf.action_dispatcher = self.old_dispatcher

    def test_decorator_registers_action(self):
        from chisurf.core.actions._decorator import action
        
        @action("test.action", schema={"value": int})
        def my_action(value: int):
            """Test action docstring."""
            return {"source_uid": "uid-1"}
            
        # Verify it's in the registry
        self.assertTrue(self.registry.has("test.action"))
        spec = self.registry.get("test.action")
        self.assertEqual(spec.name, "test.action")
        self.assertEqual(spec.schema, {"value": int})

    def test_dispatch_calls_function_and_records_history(self):
        from chisurf.core.actions._decorator import action
        
        executed = []
        @action("test.exec", schema={"x": int})
        def exec_action(x: int):
            executed.append(x)
            return {"source_uid": "src-1", "value": 123}
            
        # Call it via dispatch
        result = dispatch("test.exec", {"x": 42})
        
        self.assertEqual(executed, [42])
        self.assertEqual(result, {"source_uid": "src-1", "value": 123})
        self.assertEqual(len(self.history), 1)
        self.assertEqual(self.history[0]["action_type"], "test.exec")
        self.assertEqual(self.history[0]["payload"]["x"], 42)
        self.assertEqual(self.history[0]["source_uid"], "src-1")

    def test_direct_call_routes_through_dispatcher(self):
        from chisurf.core.actions._decorator import action
        
        executed = []
        @action("test.direct", schema={"y": int})
        def direct_action(y: int):
            executed.append(y)
            return "ok"
            
        # Call directly
        result = direct_action(y=100)
        
        self.assertEqual(executed, [100])
        self.assertEqual(result, "ok")
        self.assertEqual(len(self.history), 1)
        self.assertEqual(self.history[0]["action_type"], "test.direct")
        self.assertEqual(self.history[0]["payload"]["y"], 100)

    def test_schema_validation(self):
        from chisurf.core.actions._decorator import action
        
        @action("test.schema", schema={"val": str})
        def schema_action(val: str):
            return {}
            
        # Dispatch with wrong type
        with self.assertRaises(TypeError):
            dispatch("test.schema", {"val": 123})

    def test_builtin_actions_registered(self):
        # Import chisurf.core.actions to trigger registration of all modules
        # Since we patched chisurf.action_registry in setUp, they should register there
        import chisurf.core.actions
        from chisurf.core.actions import dataset_actions, fit_actions, model_actions, parameter_actions, project_actions
        import importlib
        
        # Reload all modules to trigger decorator-based registration again
        importlib.reload(dataset_actions)
        importlib.reload(fit_actions)
        importlib.reload(model_actions)
        importlib.reload(parameter_actions)
        importlib.reload(project_actions)
        importlib.reload(chisurf.core.actions)
        
        # Check a few random ones
        self.assertTrue(self.registry.has("dataset.add"))
        self.assertTrue(self.registry.has("fit.run"))
        self.assertTrue(self.registry.has("parameter.value"))
        self.assertTrue(self.registry.has("project.save"))

if __name__ == "__main__":
    unittest.main()
