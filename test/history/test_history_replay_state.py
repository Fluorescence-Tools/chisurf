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


if __name__ == "__main__":
    unittest.main()
