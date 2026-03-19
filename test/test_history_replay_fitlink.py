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


if __name__ == "__main__":
    unittest.main()
