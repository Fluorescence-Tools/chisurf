import pathlib
import unittest

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

from chisurf.history import OperationHistory
import chisurf
from chisurf.core.actions._infra import ActionSpec, ActionRegistry, ActionDispatcher, get_action_catalog


class TestActionDispatcher(unittest.TestCase):

    def test_execute_records_history_event(self):
        reg = ActionRegistry()
        reg.register(ActionSpec("parameter_value", schema={"parameter_name": str}))
        history = OperationHistory()
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: history)

        event = dispatcher.execute(
            name="parameter_value",
            payload={"parameter_name": "g", "new_value": 0.95},
            summary="set g",
            source_uid="p-1",
        )

        self.assertIsNotNone(event)
        rows = history.list_events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action_type"], "parameter_value")
        self.assertEqual(rows[0]["payload"]["parameter_name"], "g")
        self.assertIn("replayable", rows[0]["payload"])

    def test_debounce_suppresses_rapid_duplicates(self):
        reg = ActionRegistry()
        # debounce_ms=500 → same call within 500 ms is suppressed
        reg.register(ActionSpec("parameter_fixed", schema={"parameter_name": str}, debounce_ms=500))
        history = OperationHistory()
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: history)

        first = dispatcher.execute(
            name="parameter_fixed",
            payload={"parameter_name": "tau", "fixed": True},
            summary="fixed on",
        )
        second = dispatcher.execute(
            name="parameter_fixed",
            payload={"parameter_name": "tau", "fixed": True},
            summary="fixed on",
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(history.list_events()), 1)

    def test_debounce_zero_always_records(self):
        reg = ActionRegistry()
        reg.register(ActionSpec("project_save", debounce_ms=0))
        history = OperationHistory()
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: history)

        dispatcher.execute(name="project_save", payload={}, summary="save 1")
        dispatcher.execute(name="project_save", payload={}, summary="save 2")

        self.assertEqual(len(history.list_events()), 2)

    def test_payload_schema_validation(self):
        reg = ActionRegistry()
        reg.register(ActionSpec("parameter_link", schema={"source_parameter": str, "target_parameter": str}))
        history = OperationHistory()
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: history)

        with self.assertRaises(ValueError):
            dispatcher.execute(
                name="parameter_link",
                payload={"source_parameter": "a"},
                summary="broken",
            )

    def test_action_catalog_entries(self):
        reg = ActionRegistry()
        reg.register(ActionSpec("project_save", debounce_ms=0))

        import chisurf
        old_reg = getattr(chisurf, "action_registry", None)
        chisurf.action_registry = reg
        try:
            catalog = get_action_catalog()
            self.assertTrue(len(catalog) > 0)
            names = {entry.get("name") for entry in catalog}
            self.assertIn("project_save", names)
            first = catalog[0]
            # New style: debounce_ms instead of mcp_name/dedupe_policy
            self.assertIn("debounce_ms", first)
            self.assertEqual(first["debounce_ms"], 0)
        finally:
            chisurf.action_registry = old_reg

    def test_chisurf_action_catalog_accessor(self):
        import chisurf
        catalog_fn = getattr(chisurf, "action_catalog")
        catalog = catalog_fn()
        self.assertTrue(isinstance(catalog, list))

    def test_dispatcher_accepts_dot_and_underscore_names(self):
        reg = ActionRegistry()
        reg.register(ActionSpec("project_save"))
        history = OperationHistory()
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: history)

        event = dispatcher.execute(
            name="project.save",
            payload={"project_dir": "C:/tmp/demo"},
            summary="save project",
        )

        self.assertIsNotNone(event)
        rows = history.list_events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action_type"], "project_save")

    def test_chisurf_action_execute_accessor(self):
        executor = getattr(chisurf, "action_execute")
        event = executor(
            name="project.save",
            payload={"target_path": "C:/tmp/demo", "project_name": "demo"},
            summary="save project",
        )
        self.assertTrue(event is None or isinstance(event, dict))

    def test_record_action_falls_back_for_unregistered_type(self):
        from chisurf.core.actions._infra import record_action

        backup_history = getattr(chisurf, "history", None)
        history = OperationHistory()
        chisurf.history = history
        try:
            event = record_action(
                action_type="legacy_unregistered_event",
                summary="legacy path",
                payload={"x": 1},
            )
            self.assertIsNotNone(event)
            rows = history.list_events()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["action_type"], "legacy_unregistered_event")
        finally:
            chisurf.history = backup_history

    def test_extra_payload_keys_ignored(self):
        reg = ActionRegistry()
        def dummy_handler(a: int):
            return {"a": a}
        reg.register(ActionSpec("dummy.action", schema={"a": int}, handler=dummy_handler))
        history = OperationHistory()
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: history)

        # Dispatch with an extra key 'b' that the handler doesn't accept
        event = dispatcher.execute(
            name="dummy.action",
            payload={"a": 1, "b": 2},
            summary="testing extra keys",
        )
        self.assertIsNotNone(event)

    def test_unknown_action_returns_none(self):
        reg = ActionRegistry()
        history = OperationHistory()
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: history)

        event = dispatcher.execute(
            name="this_action_does_not_exist",
            payload={"any": "value"},
        )
        self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
