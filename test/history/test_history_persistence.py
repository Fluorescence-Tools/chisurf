import pathlib
import unittest
from unittest import mock

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

from chisurf.history import OperationHistory
import chisurf as cs
class TestHistoryPersistence(unittest.TestCase):
    def setUp(self):
        self.history = OperationHistory()
        # Add some test events
        self.history.record("test_action", "Test event 1", {"data": 1})
        self.history.record("test_action", "Test event 2", {"data": 2})
        self.history.record("test_action", "Test event 3", {"data": 3})

    def test_history_version_constants(self):
        """Test that version constants are defined."""
        self.assertEqual(self.history.HISTORY_VERSION, "1.0")
        self.assertIn("1.0", self.history.SUPPORTED_VERSIONS)

    def test_event_validation(self):
        """Test event validation functionality."""
        # Test valid event
        valid_event = {
            "event_id": "test-123",
            "timestamp": "2023-01-01T00:00:00Z",
            "action_type": "test",
            "summary": "Test event",
            "payload": {}
        }
        self.assertTrue(self.history.validate_event(valid_event))

        # Test invalid event (missing required field)
        invalid_event = {
            "event_id": "test-123",
            "timestamp": "2023-01-01T00:00:00Z",
            # Missing action_type, summary, payload
        }
        self.assertFalse(self.history.validate_event(invalid_event))

        # Test invalid event (wrong payload type)
        invalid_event2 = {
            "event_id": "test-123",
            "timestamp": "2023-01-01T00:00:00Z",
            "action_type": "test",
            "summary": "Test event",
            "payload": "not a dict"  # Should be dict
        }
        self.assertFalse(self.history.validate_event(invalid_event2))

    def test_history_integrity_validation(self):
        """Test history integrity validation."""
        # Test with valid history
        report = self.history.validate_history_integrity()
        self.assertEqual(report["total_events"], 3)
        self.assertEqual(report["valid_events"], 3)
        self.assertEqual(report["invalid_events"], [])
        self.assertFalse(report["corruption_detected"])

        # Test with corrupted history (add invalid event)
        with self.history._lock:
            self.history._events.append({"invalid": "event"})  # Missing required fields

        report = self.history.validate_history_integrity()
        self.assertEqual(report["total_events"], 4)
        self.assertEqual(report["valid_events"], 3)
        self.assertEqual(len(report["invalid_events"]), 1)
        self.assertTrue(report["corruption_detected"])

    def test_history_repair(self):
        """Test history repair functionality."""
        # Add some invalid events
        with self.history._lock:
            self.history._events.append({"invalid": "event1"})
            self.history._events.append({"invalid": "event2"})

        # Verify corruption is detected
        integrity_report = self.history.validate_history_integrity()
        self.assertTrue(integrity_report["corruption_detected"])
        self.assertEqual(len(integrity_report["invalid_events"]), 2)

        # Repair history
        repair_report = self.history.repair_history()
        self.assertTrue(repair_report["repair_successful"])
        self.assertEqual(repair_report["events_removed"], 2)
        self.assertEqual(repair_report["events_after"], 3)  # Should be back to original 3 valid events

        # Verify history is now valid
        integrity_report = self.history.validate_history_integrity()
        self.assertFalse(integrity_report["corruption_detected"])
        self.assertEqual(integrity_report["valid_events"], 3)

    def test_save_load_with_metadata(self):
        """Test saving and loading history with metadata."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_history.jsonl")
            
            # Save history
            saved_path = self.history.save_jsonl(filepath)
            self.assertTrue(saved_path.exists())
            
            # Check metadata is included
            with open(filepath, 'r') as f:
                first_line = f.readline()
                self.assertTrue(first_line.startswith("# CHISURF HISTORY METADATA:"))

    def test_load_with_version_validation(self):
        """Test loading history with version validation."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_history.jsonl")
            
            # Save history
            self.history.save_jsonl(filepath)
            
            # Load history
            load_result = self.history.load_jsonl(filepath, replace=True)
            
            # Verify load was successful
            self.assertTrue(load_result["success"])
            self.assertEqual(load_result["loaded_events"], 3)
            self.assertEqual(load_result["compatibility"], "compatible")
            self.assertEqual(load_result["file_version"], "1.0")

    def test_load_incompatible_version(self):
        """Test loading history with incompatible version."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_history_incompatible.jsonl")
            
            # Create file with incompatible version
            with open(filepath, 'w') as f:
                f.write('# CHISURF HISTORY METADATA: {"history_version": "2.0", "event_count": 1}\n')
                f.write('{"event_id": "test", "timestamp": "2023-01-01T00:00:00Z", "action_type": "test", "summary": "test", "payload": {}}\n')
            
            # Try to load
            load_result = self.history.load_jsonl(filepath, replace=True)
            
            # Should detect incompatibility but still load (for forward compatibility)
            self.assertTrue(load_result["success"])
            self.assertEqual(load_result["compatibility"], "incompatible")
            self.assertEqual(load_result["file_version"], "2.0")

    def test_corrupted_history_recovery(self):
        """Test loading and recovering from corrupted history."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corrupted.jsonl")
            
            # Create corrupted history file
            with open(filepath, 'w') as f:
                f.write('# CHISURF HISTORY METADATA: {"history_version": "1.0", "event_count": 3}\n')
                f.write('{"event_id": "test1", "timestamp": "2023-01-01T00:00:00Z", "action_type": "test", "summary": "test1", "payload": {}}\n')
                f.write('{"invalid": "event"}\n')  # Invalid event
                f.write('{"event_id": "test2", "timestamp": "2023-01-01T00:00:00Z", "action_type": "test", "summary": "test2", "payload": {}}\n')
                f.write('not valid json at all\n')  # Invalid JSON
                f.write('{"event_id": "test3", "timestamp": "2023-01-01T00:00:00Z", "action_type": "test", "summary": "test3", "payload": {}}\n')
            
            # Load should detect corruption and attempt repair
            load_result = self.history.load_jsonl(filepath, replace=True)
            
            # Should succeed but report errors
            self.assertTrue(load_result["success"])
            self.assertGreater(len(load_result["errors"]), 0)
            # Should have loaded events (including invalid ones) and then repaired
            self.assertEqual(load_result["loaded_events"], 4)  # 3 valid + 1 malformed event loaded before repair

    def test_backup_and_restore(self):
        """Test backup and restore functionality."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = os.path.join(tmpdir, "backups")
            
            # Create backup
            backup_result = self.history.create_backup(backup_dir)
            self.assertTrue(backup_result["success"])
            self.assertTrue(backup_result["backup_path"].endswith(".jsonl"))
            self.assertEqual(backup_result["event_count"], 3)
            
            # Verify backup file exists
            backup_path = backup_result["backup_path"]
            self.assertTrue(os.path.exists(backup_path))
            
            # Clear current history
            self.history.clear()
            self.assertEqual(len(self.history.list_events()), 0)
            
            # Restore from backup
            restore_result = self.history.restore_from_backup(backup_path)
            self.assertTrue(restore_result["success"])
            self.assertEqual(restore_result["loaded_events"], 3)
            
            # Verify history was restored
            self.assertEqual(len(self.history.list_events()), 3)

    def test_history_stats(self):
        """Test history statistics functionality."""
        stats = self.history.get_history_stats()
        
        self.assertEqual(stats["event_count"], 3)
        self.assertEqual(stats["checkpoint_count"], 0)
        self.assertIsNotNone(stats["oldest_event"])
        self.assertIsNotNone(stats["newest_event"])
        self.assertIn("test_action", stats["action_types"])

    def test_memory_management(self):
        """Test memory management functionality."""
        # Test initial memory limits
        limits = self.history.get_memory_limits()
        self.assertEqual(limits["max_events"], 10000)
        self.assertEqual(limits["auto_compact_threshold"], 5000)
        self.assertEqual(limits["current_event_count"], 3)

        # Test setting new limits
        self.history.set_memory_limits(max_events=5000, auto_compact_threshold=2000)
        limits = self.history.get_memory_limits()
        self.assertEqual(limits["max_events"], 5000)
        self.assertEqual(limits["auto_compact_threshold"], 2000)

        # Test memory usage estimation
        memory_usage = self.history.get_estimated_memory_usage()
        self.assertEqual(memory_usage["event_count"], 3)
        self.assertGreater(memory_usage["total_estimated_memory_bytes"], 0)

        # Test compaction (add more events first)
        for i in range(100):
            self.history.record("test_action", f"Test event {i}", {"data": i})

        # Test manual compaction
        compaction_report = self.history.compact_history(keep_recent=50)
        self.assertTrue(compaction_report["compaction_successful"])
        self.assertEqual(compaction_report["events_after"], 50)
        self.assertGreater(compaction_report["events_removed"], 0)

        # Verify compaction worked
        stats = self.history.get_history_stats()
        self.assertEqual(stats["event_count"], 50)

        # Test auto-compaction threshold
        auto_compact_report = self.history.auto_compact_if_needed()
        self.assertFalse(auto_compact_report["compaction_performed"])
        
        # Add more events to trigger auto-compaction
        while len(self.history.list_events()) < 2001:
            self.history.record("test_action", "Bulk event", {"bulk": True})
        
        auto_compact_report = self.history.auto_compact_if_needed()
        self.assertTrue(auto_compact_report.get("compaction_successful", False))


class TestBoundedCheckpoints(unittest.TestCase):
    """Stage-3: the in-memory checkpoint store is capped and evicts safely."""

    def _history(self, cap):
        # interval=1 -> a checkpoint per recorded event (after the first)
        hist = OperationHistory(checkpoint_interval=1, max_checkpoints=cap)
        # capture_fn tags each snapshot with the event count so we can identify it
        hist.set_checkpoint_capture(
            lambda: {"navigation": {"datasets": [str(len(hist.list_events()))]}}
        )
        return hist

    def test_checkpoint_count_is_capped(self):
        hist = self._history(cap=3)
        for i in range(12):
            hist.record("test_action", f"e{i}", {"i": i})
        self.assertLessEqual(hist.checkpoint_count(), 3)

    def test_eviction_keeps_earliest_and_most_recent(self):
        hist = self._history(cap=3)
        for i in range(12):
            hist.record("test_action", f"e{i}", {"i": i})
        kept = sorted(hist._checkpoints)
        # earliest retained for cheap cold-start; newest retained for cheap undo
        self.assertEqual(kept[0], min(hist._checkpoints))
        self.assertEqual(kept[0], 1)
        self.assertEqual(kept[-1], max(hist._checkpoints))

    def test_evicted_region_still_reconstructs_from_earlier_checkpoint(self):
        hist = self._history(cap=3)
        for i in range(12):
            hist.record("test_action", f"e{i}", {"i": i})
        # a target in the evicted middle still resolves to a kept checkpoint at
        # or before it, plus exactly the events needed to replay the gap
        checkpoint = hist.get_checkpoint_before(5)
        self.assertIsNotNone(checkpoint)
        self.assertLessEqual(checkpoint["event_index"], 5)
        snapshot, events = hist.get_events_from_checkpoint(5)
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(events), 5 - checkpoint["event_index"])

    def test_cap_zero_disables_eviction(self):
        hist = self._history(cap=0)
        for i in range(12):
            hist.record("test_action", f"e{i}", {"i": i})
        # no cap -> every interval checkpoint retained (legacy behaviour)
        self.assertGreater(hist.checkpoint_count(), 3)


if __name__ == "__main__":
    unittest.main()