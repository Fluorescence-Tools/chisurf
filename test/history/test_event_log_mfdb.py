"""Stage-4: MFDB-backed event log + history-as-projection (PRD-43).

Headless, sqlite-on-disk MFDB. Pins the core architectural claims:
  - the dictionary-declared ``mfdb_event_log`` table materializes;
  - ``OperationHistory.record`` dual-writes to the durable log when a DB is
    present, and is a silent no-op (history still works) when it is not;
  - **projection-equivalence**: rehydrating history from the MFDB log reproduces
    the exact same reconstructed ``DomainState`` as the in-memory log;
  - re-inserting the same ``event_id`` is idempotent (append-only restore/replay).
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest import mock

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

import chisurf.history as history
from chisurf.core.mfdb import event_log
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.result_registry import set_global_db


class _DBTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = MFDatabase(os.path.join(self._tmp.name, "t.db"))

    def tearDown(self):
        set_global_db(None)
        self._tmp.cleanup()


class TestEventLogTable(_DBTestCase):
    def test_table_materializes_from_dictionary(self):
        cols = {r[1] for r in self.db.conn.execute("PRAGMA table_info(mfdb_event_log)")}
        for expected in {
            "event_id",
            "action_type",
            "operation_type",
            "operation_id",
            "payload_json",
            "seq",
            "event_timestamp",
        }:
            self.assertIn(expected, cols)


class TestAppendRead(_DBTestCase):
    def test_append_then_read_roundtrip(self):
        ev = {
            "event_id": "e1",
            "action_type": "dataset.add",
            "summary": "add ds",
            "payload": {"loaded_names": ["ds0"], "loaded_uids": ["u0"]},
            "source_uid": None,
            "target_uid": None,
            "timestamp": "2026-06-27T00:00:00+00:00",
        }
        self.assertTrue(event_log.append_event(ev, db=self.db))
        out = event_log.read_events(db=self.db)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["action_type"], "dataset.add")
        self.assertEqual(out[0]["payload"], {"loaded_names": ["ds0"], "loaded_uids": ["u0"]})

    def test_reinsert_same_event_id_is_idempotent(self):
        ev = {
            "event_id": "dup",
            "action_type": "fit.add",
            "summary": "x",
            "payload": {},
            "timestamp": "2026-06-27T00:00:00+00:00",
        }
        self.assertTrue(event_log.append_event(ev, db=self.db))
        event_log.append_event(ev, db=self.db)  # second insert ignored
        self.assertEqual(len(event_log.read_events(db=self.db)), 1)


class TestRecordDualWrite(_DBTestCase):
    def test_record_writes_to_mfdb_when_db_present(self):
        set_global_db(self.db)
        hist = history.OperationHistory()
        hist.record("dataset.add", "add", {"loaded_names": ["ds0"], "loaded_uids": ["u0"]})
        rows = event_log.read_events(db=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action_type"], "dataset.add")

    def test_projection_equivalence(self):
        """History rehydrated from MFDB reconstructs the identical DomainState."""
        set_global_db(self.db)
        live = history.OperationHistory()
        live.record("dataset.add", "add ds", {"loaded_names": ["ds0"], "loaded_uids": ["u0"]})
        live.record("fit.add", "add fit", {"fit_group_name": "fit0", "fit_uid": "f0"})
        live.record(
            "parameter.value",
            "set tau",
            {"fit_group": "f0", "local_fit": "local_0", "parameter_name": "tau", "new_value": 4.0},
        )

        # rehydrate a fresh history purely from the durable log
        restored = history.OperationHistory()
        restored.load_events(event_log.read_events(db=self.db), replace=True)

        live_state = history.build_target_state(None, live.list_events())
        restored_state = history.build_target_state(None, restored.list_events())
        self.assertEqual(restored_state, live_state)
        # and the events themselves match in order/content where it matters
        self.assertEqual(
            [e["action_type"] for e in restored.list_events()],
            [e["action_type"] for e in live.list_events()],
        )


class TestProjectArchiveRoundTrip(_DBTestCase):
    def test_archive_then_restore_preserves_history(self):
        from chisurf.core.mfdb.project_archiver import (
            archive_project_to_mfdb,
            restore_project_from_artifacts,
        )

        events = [
            {
                "event_id": "a1",
                "action_type": "dataset.add",
                "summary": "add",
                "payload": {"loaded_names": ["ds0"]},
                "source_uid": None,
                "target_uid": None,
                "timestamp": "2026-06-27T00:00:00+00:00",
            },
            {
                "event_id": "a2",
                "action_type": "fit.add",
                "summary": "fit",
                "payload": {"fit_group_name": "fit0", "fit_uid": "f0"},
                "source_uid": None,
                "target_uid": None,
                "timestamp": "2026-06-27T00:00:01+00:00",
            },
        ]
        payload = {
            "meta": {"name": "demo"},
            "datasets": {},
            "fits": [],
            "extra": {"history_events": events},
        }
        archive_project_to_mfdb(
            db=self.db,
            project_payload=payload,
            version_id="ver_1",
            project_id="proj_1",
            version_number=1,
        )
        # archive stamped the events with the project_id, so a project-scoped read
        # returns them in order — this is what restore reads back into cs.history
        got = event_log.read_events(project_id="proj_1", db=self.db)
        self.assertEqual([e["event_id"] for e in got], ["a1", "a2"])
        self.assertEqual([e["action_type"] for e in got], ["dataset.add", "fit.add"])
        # events not belonging to this project are not returned
        self.assertEqual(event_log.read_events(project_id="proj_other", db=self.db), [])
        # restore_project_from_artifacts wires history into extra.history_events
        # for any project that has output artifacts (empty project -> None here)
        self.assertIsNone(restore_project_from_artifacts(self.db, "ver_1"))


class TestNoDbRegression(_DBTestCase):
    def test_record_is_noop_without_db_and_history_still_works(self):
        # Force genuinely-no-MFDB: on a dev box _get_global_db falls back to the
        # configured user database, so patch it to None to exercise offline mode
        # (and to avoid writing to the real user DB).
        with mock.patch("chisurf.core.mfdb.result_registry._get_global_db", return_value=None):
            hist = history.OperationHistory()
            ev = hist.record("dataset.add", "add", {"loaded_names": ["ds0"]})
            # in-memory history fully functional
            self.assertEqual(len(hist.list_events()), 1)
            self.assertEqual(ev["action_type"], "dataset.add")
            # mfdb-sourced read is empty (graceful), not an error
            self.assertEqual(hist.list_events(source="mfdb"), [])


if __name__ == "__main__":
    unittest.main()
