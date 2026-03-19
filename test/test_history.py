import pathlib
import tempfile
import unittest

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

from chisurf.history import OperationHistory


class TestHistory(unittest.TestCase):

    def test_record_and_subscribe(self):
        history = OperationHistory()
        seen = []

        def callback(event):
            seen.append(event)

        history.subscribe(callback)
        event = history.record(
            action_type="parameter_value",
            summary="set value",
            payload={"name": "tau", "value": 1.23},
            source_uid="fit-1",
            target_uid="param-1",
        )

        self.assertEqual(len(history.list_events()), 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(event["action_type"], "parameter_value")
        self.assertEqual(event["payload"]["name"], "tau")
        self.assertEqual(event["source_uid"], "fit-1")
        self.assertEqual(event["target_uid"], "param-1")

    def test_save_load_jsonl(self):
        history = OperationHistory()
        history.record("a", "first", {"i": 1})
        history.record("b", "second", {"i": 2})

        with tempfile.TemporaryDirectory() as tmpdir:
            target = pathlib.Path(tmpdir) / "history.jsonl"
            history.save_jsonl(target)

            other = OperationHistory()
            result = other.load_jsonl(target, replace=True)

            self.assertEqual(result["loaded_events"], 2)
            self.assertEqual(len(other.list_events()), 2)
            self.assertEqual(other.list_events()[0]["action_type"], "a")
            self.assertEqual(other.list_events()[1]["summary"], "second")

    def test_replay(self):
        history = OperationHistory()
        history.record("load", "load dataset", {"dataset": "d1"})
        history.record("fit", "run fit", {"fit": "f1"})
        history.record("unknown", "skip me", {})

        calls = []

        def on_load(event):
            calls.append(("load", event["payload"]["dataset"]))

        def on_fit(event):
            calls.append(("fit", event["payload"]["fit"]))

        report = history.replay(
            handlers={
                "load": on_load,
                "fit": on_fit,
            },
            stop_on_error=True,
        )

        self.assertEqual(report["total"], 3)
        self.assertEqual(report["replayed"], 2)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(calls, [("load", "d1"), ("fit", "f1")])

    def test_checkpoint_creation(self):
        history = OperationHistory(checkpoint_interval=5)
        snapshots = []

        def capture_fn():
            snapshot = {"counter": len(snapshots)}
            snapshots.append(snapshot)
            return snapshot

        history.set_checkpoint_capture(capture_fn)

        for i in range(15):
            history.record("test", f"event {i}", {"index": i})

        self.assertEqual(history.checkpoint_count(), 2)
        self.assertEqual(len(snapshots), 2)

    def test_get_checkpoint_before(self):
        history = OperationHistory(checkpoint_interval=10)

        def capture_fn():
            return {"state": "captured"}

        history.set_checkpoint_capture(capture_fn)

        for i in range(25):
            history.record("test", f"event {i}", {"index": i})

        checkpoint = history.get_checkpoint_before(15)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint["event_index"], 10)
        self.assertEqual(checkpoint["snapshot"]["state"], "captured")

        checkpoint = history.get_checkpoint_before(5)
        self.assertIsNone(checkpoint)

    def test_get_events_from_checkpoint(self):
        history = OperationHistory(checkpoint_interval=10)

        def capture_fn():
            return {"state": "captured"}

        history.set_checkpoint_capture(capture_fn)

        for i in range(25):
            history.record("test", f"event {i}", {"index": i})

        snapshot, events = history.get_events_from_checkpoint(15)
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0]["payload"]["index"], 11)

        snapshot, events = history.get_events_from_checkpoint(5)
        self.assertIsNone(snapshot)
        self.assertEqual(len(events), 6)

    def test_clear_clears_checkpoints(self):
        history = OperationHistory(checkpoint_interval=5)

        def capture_fn():
            return {"state": "captured"}

        history.set_checkpoint_capture(capture_fn)

        for i in range(10):
            history.record("test", f"event {i}", {"index": i})

        self.assertEqual(history.checkpoint_count(), 1)
        history.clear()
        self.assertEqual(history.checkpoint_count(), 0)

    def test_load_jsonl_clears_checkpoints(self):
        history = OperationHistory(checkpoint_interval=5)

        def capture_fn():
            return {"state": "captured"}

        history.set_checkpoint_capture(capture_fn)

        for i in range(10):
            history.record("test", f"event {i}", {"index": i})

        self.assertEqual(history.checkpoint_count(), 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = pathlib.Path(tmpdir) / "history.jsonl"
            history.save_jsonl(target)
            history.load_jsonl(target, replace=True)

        self.assertEqual(history.checkpoint_count(), 0)


if __name__ == "__main__":
    unittest.main()
