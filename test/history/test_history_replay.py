# Consolidated test file: test_history_replay.py


# --- FROM test_history_replay.py ---

# --- FROM test_history_persistence.py ---
import pathlib
import unittest
from unittest import mock

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

from chisurf.history import OperationHistory
import chisurf


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



# --- FROM test_history_replay_state.py ---
import pathlib
import unittest

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

from chisurf.history import replay as history_replay


class TestHistoryReplayState(unittest.TestCase):

    def test_reconstruct_navigation_state(self):
        events = [
            {"action_type": "dataset_add", "payload": {"loaded_names": ["d1"]}},
            {"action_type": "dataset_add", "payload": {"loaded_names": ["d2"]}},
            {"action_type": "fit_add", "source_uid": "fit-u1", "payload": {"fit_group_name": "f1"}},
            {"action_type": "fit_add", "source_uid": "fit-u2", "payload": {"fit_group_name": "f2"}},
            {"action_type": "fit_close", "source_uid": "fit-u2", "payload": {"fit_name": "f2"}},
        ]
        state = history_replay.reconstruct_navigation_state(events)
        self.assertEqual(state["selected_dataset"], "d2")
        self.assertEqual(state["selected_fit"], "f1")
        self.assertEqual(state["selected_fit_uid"], "fit-u1")
        self.assertIn("d1", state["datasets"])
        self.assertIn("f1", state["fits"])

    def test_reconstruct_navigation_state_dataset_ungroup(self):
        events = [
            {
                "action_type": "dataset_add",
                "payload": {"loaded_names": ["a", "b"], "loaded_uids": ["u-a", "u-b"]},
            },
            {
                "action_type": "dataset_group",
                "payload": {"group_name": "grp", "group_uid": "u-g"},
            },
            {
                "action_type": "dataset_ungroup",
                "payload": {
                    "group_names": ["grp"],
                    "group_uids": ["u-g"],
                    "expanded_names": ["a", "b"],
                    "expanded_uids": ["u-a", "u-b"],
                },
            },
        ]
        state = history_replay.reconstruct_navigation_state(events)
        self.assertNotIn("grp", state["datasets"])
        self.assertIn("a", state["datasets"])
        self.assertIn("b", state["datasets"])
        self.assertEqual(state["selected_dataset"], "b")
        self.assertEqual(state["selected_dataset_uid"], "u-b")

    def test_reconstruct_navigation_state_prefers_dataset_uid(self):
        events = [
            {
                "action_type": "dataset_add",
                "payload": {
                    "loaded_names": ["same", "same"],
                    "loaded_uids": ["uid-1", "uid-2"],
                },
            },
            {
                "action_type": "dataset_remove",
                "payload": {
                    "removed_names": ["same"],
                    "removed_uids": ["uid-1"],
                },
            },
        ]
        state = history_replay.reconstruct_navigation_state(events)
        self.assertEqual(state["selected_dataset_uid"], "uid-2")

    def test_reconstruct_navigation_state_tracks_fit_uid(self):
        events = [
            {"action_type": "fit_add", "source_uid": "fit-a", "payload": {"fit_group_name": "A"}},
            {"action_type": "fit_run_start", "source_uid": "fit-a", "payload": {"fit_name": "A"}},
            {"action_type": "fit_add", "source_uid": "fit-b", "payload": {"fit_group_name": "B"}},
            {"action_type": "fit_run_finish", "source_uid": "fit-b", "payload": {"fit_name": "B"}},
        ]
        state = history_replay.reconstruct_navigation_state(events)
        self.assertEqual(state["selected_fit"], "B")
        self.assertEqual(state["selected_fit_uid"], "fit-b")

    def test_reconstruct_parameter_state(self):
        events = [
            {
                "action_type": "parameter_value",
                "payload": {
                    "fit_group": "fg",
                    "local_fit": "l1",
                    "parameter_name": "tau",
                    "new_value": 2.0,
                },
            },
            {
                "action_type": "parameter_fixed",
                "payload": {
                    "fit_group": "fg",
                    "local_fit": "l1",
                    "parameter_name": "tau",
                    "fixed": True,
                },
            },
            {
                "action_type": "parameter_link",
                "payload": {
                    "source_fit_group": "fg",
                    "source_local_fit": "l1",
                    "source_parameter": "tau",
                    "target_fit_group": "fg2",
                    "target_local_fit": "l2",
                    "target_parameter": "tau",
                },
            },
            {
                "action_type": "parameter_unlink",
                "payload": {
                    "fit_group": "fg",
                    "local_fit": "l1",
                    "parameter_name": "tau",
                },
            },
        ]
        state = history_replay.reconstruct_parameter_state(events)
        key = ("fg", "l1", "tau")
        self.assertIn(key, state)
        self.assertEqual(state[key]["value"], 2.0)
        self.assertEqual(state[key]["fixed"], True)
        self.assertIsNone(state[key]["link"])

    def test_reconstruct_parameter_state_link_uid_metadata(self):
        events = [
            {
                "action_type": "parameter_link",
                "payload": {
                    "source_fit_group": "fg",
                    "source_local_fit": "l1",
                    "source_parameter": "tau",
                    "source_fit_uid": "fit-u-1",
                    "source_local_fit_uid": "local-u-1",
                    "source_parameter_uid": "param-u-1",
                    "target_fit_group": "fg2",
                    "target_local_fit": "l2",
                    "target_parameter": "tau",
                    "target_fit_uid": "fit-u-2",
                    "target_local_fit_uid": "local-u-2",
                    "target_parameter_uid": "param-u-2",
                },
            }
        ]
        state = history_replay.reconstruct_parameter_state(events)
        key = ("fg", "l1", "tau")
        self.assertEqual(state[key]["source_fit_uid"], "fit-u-1")
        self.assertEqual(state[key]["source_local_fit_uid"], "local-u-1")
        self.assertEqual(state[key]["source_parameter_uid"], "param-u-1")
        self.assertEqual(state[key]["link_uid"], ("fit-u-2", "local-u-2", "param-u-2"))

    def test_touched_parameter_keys(self):
        events = [
            {
                "action_type": "parameter_link",
                "payload": {
                    "source_fit_group": "fg",
                    "source_local_fit": "l1",
                    "source_parameter": "tau",
                    "target_fit_group": "fg2",
                    "target_local_fit": "l2",
                    "target_parameter": "tau",
                },
            },
            {
                "action_type": "parameter_value",
                "payload": {
                    "fit_group": "fg",
                    "local_fit": "l1",
                    "parameter_name": "tau",
                    "new_value": 1.0,
                },
            },
        ]
        keys = history_replay.touched_parameter_keys(events)
        self.assertIn(("fg", "l1", "tau"), keys)

    def test_fit_run_snapshots_restore_parameter_state(self):
        events = [
            {
                "action_type": "fit_run_start",
                "payload": {
                    "parameter_snapshot_before": [
                        {
                            "fit_group": "fg",
                            "local_fit": "l1",
                            "parameter_name": "tau",
                            "value": 1.5,
                            "fixed": False,
                            "bounds_on": True,
                            "lower": 0.1,
                            "upper": 10.0,
                        }
                    ]
                },
            },
            {
                "action_type": "fit_run_finish",
                "payload": {
                    "parameter_snapshot_after": [
                        {
                            "fit_group": "fg",
                            "local_fit": "l1",
                            "parameter_name": "tau",
                            "value": 3.0,
                            "fixed": True,
                            "bounds_on": False,
                            "lower": 0.5,
                            "upper": 5.0,
                        }
                    ]
                },
            },
        ]

        before_only = history_replay.reconstruct_parameter_state(events[:1])
        after_all = history_replay.reconstruct_parameter_state(events)

        key = ("fg", "l1", "tau")
        self.assertEqual(before_only[key]["value"], 1.5)
        self.assertEqual(before_only[key]["fixed"], False)
        self.assertEqual(before_only[key]["bounds_on"], True)
        self.assertEqual(before_only[key]["bounds"], (0.1, 10.0))

        self.assertEqual(after_all[key]["value"], 3.0)
        self.assertEqual(after_all[key]["fixed"], True)
        self.assertEqual(after_all[key]["bounds_on"], False)
        self.assertEqual(after_all[key]["bounds"], (0.5, 5.0))

    def test_reconstruct_fit_range_state(self):
        events = [
            {
                "action_type": "fit_range_set",
                "payload": {
                    "fit_group": "fg",
                    "xmin": 10,
                    "xmax": 50,
                },
            },
            {
                "action_type": "fit_run_start",
                "payload": {
                    "fit_range_snapshot_before": [
                        {"fit_group": "fg", "local_fit": "l1", "xmin": 10, "xmax": 50}
                    ]
                },
            },
            {
                "action_type": "fit_run_finish",
                "payload": {
                    "fit_range_snapshot_after": [
                        {"fit_group": "fg", "local_fit": "l1", "xmin": 12, "xmax": 48}
                    ]
                },
            },
        ]

        state_before_finish = history_replay.reconstruct_fit_range_state(events[:2])
        state_after_finish = history_replay.reconstruct_fit_range_state(events)

        self.assertEqual(state_before_finish["fg"]["xmin"], 10)
        self.assertEqual(state_before_finish["fg"]["xmax"], 50)
        self.assertEqual(state_after_finish["fg"]["xmin"], 12)
        self.assertEqual(state_after_finish["fg"]["xmax"], 48)

    def test_reconstruct_setup_state(self):
        events = [
            {"action_type": "experiment_set", "payload": {"name": "TCSPC"}},
            {"action_type": "setup_select", "payload": {"name": "PDA"}},
            {
                "action_type": "setup_params_set",
                "payload": {
                    "params": {
                        "minimum_time_window_length": 0.002,
                        "noise_model.weight_type": "suren",
                    }
                },
            },
            {
                "action_type": "setup_params_set",
                "payload": {
                    "params": {
                        "noise_model.weight_type": "uniform",
                    }
                },
            },
        ]

        state = history_replay.reconstruct_setup_state(events)
        self.assertEqual(state["experiment"], "TCSPC")
        self.assertEqual(state["setup"], "PDA")
        self.assertEqual(state["params"]["minimum_time_window_length"], 0.002)
        self.assertEqual(state["params"]["noise_model.weight_type"], "uniform")

    def test_snapshot_to_replay_state(self):
        snapshot = {
            "navigation": {
                "datasets": ["d1", "d2"],
                "dataset_uids": ["u1", "u2"],
                "fits": ["f1"],
                "fit_uids": ["f1-u"],
                "selected_dataset": "d2",
                "selected_dataset_uid": "u2",
                "selected_fit": "f1",
                "selected_fit_uid": "f1-u",
            },
            "parameters": {
                "fg/l1/tau": {
                    "fit_group": "fg",
                    "fit_group_uid": "fg-u",
                    "local_fit": "l1",
                    "local_fit_uid": "l1-u",
                    "parameter_name": "tau",
                    "parameter_uid": "tau-u",
                    "value": 2.5,
                    "fixed": True,
                    "bounds_on": True,
                    "bounds": [0.1, 10.0],
                    "link": {
                        "fit_group": "fg2",
                        "local_fit": "l2",
                        "parameter_name": "tau",
                    },
                },
            },
            "fit_ranges": {
                "fg": {"xmin": 10, "xmax": 100},
            },
            "setup": {
                "experiment": "TCSPC",
                "setup": "PDA",
            },
        }

        state = history_replay.snapshot_to_replay_state(snapshot)

        self.assertEqual(state["navigation"]["datasets"], ["d1", "d2"])
        self.assertEqual(state["navigation"]["selected_dataset"], "d2")
        self.assertEqual(state["navigation"]["selected_fit"], "f1")

        param_key = ("fg", "l1", "tau")
        self.assertIn(param_key, state["parameters"])
        self.assertEqual(state["parameters"][param_key]["value"], 2.5)
        self.assertEqual(state["parameters"][param_key]["fixed"], True)
        self.assertEqual(state["parameters"][param_key]["bounds"], (0.1, 10.0))
        self.assertEqual(state["parameters"][param_key]["link"], ("fg2", "l2", "tau"))
        self.assertEqual(state["parameters"][param_key]["source_fit_uid"], "fg-u")

        self.assertEqual(state["fit_ranges"]["fg"]["xmin"], 10)
        self.assertEqual(state["setup"]["experiment"], "TCSPC")







    def test_reconstruct_model_state(self):
        events = [
            {
                "action_type": "model_add_component",
                "source_uid": "fit-u1",
                "payload": {
                    "component_name": "gaussian1",
                },
            },
            {
                "action_type": "model_set_correction",
                "source_uid": "fit-u1",
                "payload": {
                    "correction_type": "pileup",
                    "value": 0.05,
                },
            },
            {
                "action_type": "model_change_irf",
                "source_uid": "fit-u1",
                "payload": {
                    "irf_idx": 0,
                    "irf_name": "measured",
                },
            },
            {
                "action_type": "model_remove_component",
                "source_uid": "fit-u1",
                "payload": {
                    "component_name": "gaussian1",
                },
            },
        ]

        state = history_replay.reconstruct_model_state(events)
        
        # Check that the fit group was tracked
        self.assertIn("fit-u1", state)
        
        fg_state = state["fit-u1"]
        self.assertEqual(fg_state["fit_group_uid"], "fit-u1")
        
        # Check that local fits were tracked (default local_0)
        self.assertIn("local_0", fg_state["local_fits"])
        
        local_state = fg_state["local_fits"]["local_0"]
        
        # Check components - the remove should overwrite the add
        self.assertEqual(len(local_state["components"]), 1)
        self.assertEqual(local_state["components"][0]["name"], "gaussian1")
        self.assertEqual(local_state["components"][0]["action"], "remove")
        
        # Check config
        self.assertIn("correction_pileup", local_state["config"])
        self.assertEqual(local_state["config"]["correction_pileup"], 0.05)
        self.assertIn("irf_0", local_state["config"])
        self.assertEqual(local_state["config"]["irf_0"], "measured")

    def test_reconstruct_model_state_multiple_fits(self):
        events = [
            {
                "action_type": "model_add_component",
                "source_uid": "fit-u1",
                "payload": {"component_name": "comp1"},
            },
            {
                "action_type": "model_add_component",
                "source_uid": "fit-u2", 
                "payload": {"component_name": "comp2"},
            },
        ]

        state = history_replay.reconstruct_model_state(events)
        
        # Check that both fit groups were tracked
        self.assertIn("fit-u1", state)
        self.assertIn("fit-u2", state)
        
        # Check components in each fit
        self.assertEqual(len(state["fit-u1"]["local_fits"]["local_0"]["components"]), 1)
        self.assertEqual(state["fit-u1"]["local_fits"]["local_0"]["components"][0]["name"], "comp1")
        
        self.assertEqual(len(state["fit-u2"]["local_fits"]["local_0"]["components"]), 1)
        self.assertEqual(state["fit-u2"]["local_fits"]["local_0"]["components"][0]["name"], "comp2")



# --- FROM test_history_replay_fitlink.py ---
import pathlib
import unittest

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

from chisurf.history import OperationHistory


def _make_state_handlers(state):
    def on_fit_add(event):
        payload = event.get("payload", {})
        fit_uid = str(event.get("source_uid") or payload.get("fit_uid") or "")
        fit_name = str(payload.get("fit_group_name") or payload.get("fit_name") or fit_uid)
        if not fit_uid:
            return
        state["fits"][fit_uid] = {
            "name": fit_name,
            "parameters": {},
        }

    def on_parameter_value(event):
        payload = event.get("payload", {})
        fit_uid = str(payload.get("fit_uid") or event.get("source_uid") or "")
        param_name = str(payload.get("parameter_name") or "")
        value = payload.get("new_value", payload.get("value"))
        if fit_uid and param_name and fit_uid in state["fits"]:
            state["fits"][fit_uid]["parameters"][param_name] = value

    def on_parameter_link(event):
        payload = event.get("payload", {})
        src_fit_uid = str(payload.get("source_fit_uid") or payload.get("fit_uid") or event.get("source_uid") or "")
        src_param = str(payload.get("source_parameter") or payload.get("parameter_name") or "")
        tgt_fit_uid = str(payload.get("target_fit_uid") or "")
        tgt_param = str(payload.get("target_parameter") or "")
        if src_fit_uid and src_param and tgt_fit_uid and tgt_param:
            state["links"][f"{src_fit_uid}:{src_param}"] = f"{tgt_fit_uid}:{tgt_param}"

    def on_parameter_unlink(event):
        payload = event.get("payload", {})
        src_fit_uid = str(payload.get("fit_uid") or event.get("source_uid") or "")
        src_param = str(payload.get("parameter_name") or "")
        key = f"{src_fit_uid}:{src_param}"
        state["links"].pop(key, None)

    def on_fit_close(event):
        fit_uid = str(event.get("source_uid") or "")
        if not fit_uid:
            return
        state["fits"].pop(fit_uid, None)
        linked_keys = [k for k in list(state["links"].keys()) if k.startswith(f"{fit_uid}:")]
        for key in linked_keys:
            state["links"].pop(key, None)
        target_keys = [k for k, v in state["links"].items() if str(v).startswith(f"{fit_uid}:")]
        for key in target_keys:
            state["links"].pop(key, None)

    return {
        "fit_add": on_fit_add,
        "parameter_value": on_parameter_value,
        "parameter_link": on_parameter_link,
        "parameter_unlink": on_parameter_unlink,
        "fit_close": on_fit_close,
    }


class TestHistoryReplayFitLink(unittest.TestCase):

    def test_replay_equivalence_for_fit_and_link_actions(self):
        events = [
            {
                "action_type": "fit_add",
                "source_uid": "fit-A",
                "payload": {"fit_uid": "fit-A", "fit_group_name": "Fit A"},
            },
            {
                "action_type": "fit_add",
                "source_uid": "fit-B",
                "payload": {"fit_uid": "fit-B", "fit_group_name": "Fit B"},
            },
            {
                "action_type": "parameter_value",
                "source_uid": "fit-A",
                "payload": {"fit_uid": "fit-A", "parameter_name": "tau", "new_value": 2.5},
            },
            {
                "action_type": "parameter_link",
                "source_uid": "fit-A",
                "payload": {
                    "source_fit_uid": "fit-A",
                    "source_parameter": "tau",
                    "target_fit_uid": "fit-B",
                    "target_parameter": "tau",
                },
            },
            {
                "action_type": "parameter_unlink",
                "source_uid": "fit-A",
                "payload": {"fit_uid": "fit-A", "parameter_name": "tau"},
            },
            {
                "action_type": "fit_close",
                "source_uid": "fit-B",
                "payload": {},
            },
        ]

        expected_state = {"fits": {}, "links": {}}
        expected_handlers = _make_state_handlers(expected_state)
        for event in events:
            expected_handlers[event["action_type"]](event)

        replay_state = {"fits": {}, "links": {}}
        replay_handlers = _make_state_handlers(replay_state)
        history = OperationHistory()
        report = history.replay(replay_handlers, events=events, stop_on_error=True)

        self.assertEqual(report["total"], len(events))
        self.assertEqual(report["replayed"], len(events))
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(len(report["errors"]), 0)
        self.assertDictEqual(replay_state, expected_state)

    def test_replay_reports_handler_errors(self):
        events = [
            {"action_type": "fit_add", "source_uid": "fit-A", "payload": {"fit_uid": "fit-A"}},
            {"action_type": "parameter_value", "source_uid": "fit-A", "payload": {"fit_uid": "fit-A", "parameter_name": "tau", "new_value": 1.0}},
        ]

        state = {"fits": {}, "links": {}}
        handlers = _make_state_handlers(state)

        def bad_handler(event):
            _ = event
            raise RuntimeError("boom")

        handlers["parameter_value"] = bad_handler

        history = OperationHistory()
        report = history.replay(handlers, events=events, stop_on_error=False)

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["replayed"], 1)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(len(report["errors"]), 1)

    def test_replay_equivalence_for_dataset_actions(self):
        events = [
            {
                "action_type": "dataset_add",
                "payload": {"loaded_names": ["dA"], "loaded_count": 1},
            },
            {
                "action_type": "dataset_add",
                "payload": {"loaded_names": ["dB", "dC"], "loaded_count": 2},
            },
            {
                "action_type": "dataset_group",
                "payload": {"group_name": "Data-Group", "group_size": 2},
            },
            {
                "action_type": "dataset_remove",
                "payload": {"removed_names": ["dA"], "removed_count": 1},
            },
        ]

        def make_handlers(state):
            def on_add(event):
                names = list(event.get("payload", {}).get("loaded_names", []))
                state["datasets"].extend([str(n) for n in names])

            def on_group(event):
                payload = event.get("payload", {})
                state["groups"].append({
                    "name": str(payload.get("group_name", "Data-Group")),
                    "size": int(payload.get("group_size", 0)),
                })

            def on_remove(event):
                names = set(str(n) for n in event.get("payload", {}).get("removed_names", []))
                state["datasets"] = [d for d in state["datasets"] if d not in names]

            return {
                "dataset_add": on_add,
                "dataset_group": on_group,
                "dataset_remove": on_remove,
            }

        expected_state = {"datasets": [], "groups": []}
        for event in events:
            make_handlers(expected_state)[event["action_type"]](event)

        replay_state = {"datasets": [], "groups": []}
        history = OperationHistory()
        report = history.replay(make_handlers(replay_state), events=events)

        self.assertEqual(report["total"], 4)
        self.assertEqual(report["replayed"], 4)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(report["errors"], [])
        self.assertDictEqual(replay_state, expected_state)

    def test_replay_equivalence_for_fit_run_lifecycle(self):
        events = [
            {
                "action_type": "fit_run_start",
                "source_uid": "fit-A",
                "payload": {"fit_name": "Fit A", "local_first": True, "n_steps": 1000, "n_runs": 5},
            },
            {
                "action_type": "fit_run_finish",
                "source_uid": "fit-A",
                "payload": {"fit_name": "Fit A", "success": True, "elapsed_ms": 1234, "result_count": 3},
            },
        ]

        def make_handlers(state):
            def on_start(event):
                fit_uid = str(event.get("source_uid") or "")
                payload = event.get("payload", {})
                state["runs"][fit_uid] = {
                    "status": "running",
                    "fit_name": str(payload.get("fit_name", "")),
                    "n_steps": int(payload.get("n_steps", 0)),
                    "n_runs": int(payload.get("n_runs", 0)),
                }

            def on_finish(event):
                fit_uid = str(event.get("source_uid") or "")
                payload = event.get("payload", {})
                state["runs"][fit_uid] = {
                    "status": "finished" if bool(payload.get("success", False)) else "aborted",
                    "fit_name": str(payload.get("fit_name", "")),
                    "elapsed_ms": int(payload.get("elapsed_ms", 0)),
                    "result_count": int(payload.get("result_count", 0)),
                }

            return {
                "fit_run_start": on_start,
                "fit_run_finish": on_finish,
                "fit_run_abort": on_finish,
            }

        expected_state = {"runs": {}}
        for event in events:
            make_handlers(expected_state)[event["action_type"]](event)

        replay_state = {"runs": {}}
        history = OperationHistory()
        report = history.replay(make_handlers(replay_state), events=events)

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["replayed"], 2)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(report["errors"], [])
        self.assertDictEqual(replay_state, expected_state)

    def test_replay_equivalence_for_project_lifecycle_actions(self):
        events = [
            {
                "action_type": "project_save",
                "payload": {"project_dir": "C:/tmp/p1", "project_name": "p1"},
            },
            {
                "action_type": "project_close",
                "payload": {"current_project_dir": "C:/tmp/p1"},
            },
            {
                "action_type": "app_reinitialize_start",
                "payload": {"has_main_window": True},
            },
            {
                "action_type": "app_reinitialize_finish",
                "payload": {"has_main_window": True},
            },
            {
                "action_type": "project_load",
                "payload": {"project_path": "C:/tmp/p1"},
            },
        ]

        def make_handlers(state):
            def on_project_save(event):
                payload = event.get("payload", {})
                state["project_path"] = str(payload.get("project_dir", ""))
                state["saved"] = True

            def on_project_close(_event):
                state["closed"] = True

            def on_reinit_start(_event):
                state["reinitializing"] = True

            def on_reinit_finish(_event):
                state["reinitializing"] = False
                state["reinitialized"] = True

            def on_project_load(event):
                payload = event.get("payload", {})
                state["loaded_path"] = str(payload.get("project_path", ""))
                state["loaded"] = True

            return {
                "project_save": on_project_save,
                "project_close": on_project_close,
                "app_reinitialize_start": on_reinit_start,
                "app_reinitialize_finish": on_reinit_finish,
                "project_load": on_project_load,
            }

        expected_state = {
            "project_path": "",
            "saved": False,
            "closed": False,
            "reinitializing": False,
            "reinitialized": False,
            "loaded_path": "",
            "loaded": False,
        }
        expected_handlers = make_handlers(expected_state)
        for event in events:
            expected_handlers[event["action_type"]](event)

        replay_state = {
            "project_path": "",
            "saved": False,
            "closed": False,
            "reinitializing": False,
            "reinitialized": False,
            "loaded_path": "",
            "loaded": False,
        }
        history = OperationHistory()
        report = history.replay(make_handlers(replay_state), events=events)

        self.assertEqual(report["total"], 5)
        self.assertEqual(report["replayed"], 5)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(report["errors"], [])
        self.assertDictEqual(replay_state, expected_state)


