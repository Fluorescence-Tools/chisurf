"""PRD-11: register_operation — uniform operation-node recording.

Records a data operation as a node (typed inputs/outputs + derived_from edges +
role-indexed parameters), validated against the operation type's .dic schema.
"""

from __future__ import annotations

import os

import pytest

from chisurf.core.mfdb.operation_parameters import OperationParameterError
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.result_registry import (
    register_operation,
    register_raw_measurement,
    set_global_db,
)


def _db(tmp_path) -> MFDatabase:
    return MFDatabase(os.path.join(tmp_path, "ops.db"))


def _raw(db, tmp_path, name="m.ptu") -> str:
    f = tmp_path / name
    f.write_bytes(b"\x00\x01")
    return register_raw_measurement(str(f), db=db)


def test_register_operation_records_node_and_links(tmp_path):
    db = _db(tmp_path)
    try:
        src = _raw(db, tmp_path)
        assert src
        op_id = register_operation(
            operation_type="microtime_shift",
            inputs=[src],
            outputs=[],
            parameters={"global_shift": 5},
            db=db,
        )
        assert op_id
        # the operation exists with the right type
        row = db.conn.execute(
            "SELECT operation_type FROM mfdb_operation WHERE operation_id = ?", (op_id,)
        ).fetchone()
        assert row[0] == "microtime_shift"
        # an input port link was recorded
        n_inputs = db.conn.execute(
            "SELECT COUNT(*) FROM mfdb_operation_artifact WHERE operation_id=? AND direction='input'",
            (op_id,),
        ).fetchone()[0]
        assert n_inputs == 1
    finally:
        set_global_db(None)
        db.close()


def test_register_operation_role_indexed_parameters(tmp_path):
    db = _db(tmp_path)
    try:
        op_id = register_operation(
            operation_type="microtime_shift",
            parameters={
                "global_shift": 0,
                # repeatable per-channel shift -> one row per channel, role-indexed
                "shift": [
                    {"value": 5, "role": "0"},
                    {"value": 3, "role": "1"},
                ],
            },
            db=db,
        )
        rows = db.conn.execute(
            "SELECT name, value, role FROM mfdb_parameter WHERE operation_id=? AND name='shift' ORDER BY role",
            (op_id,),
        ).fetchall()
        assert [(r[1], r[2]) for r in rows] == [(5.0, "0"), (3.0, "1")]
    finally:
        set_global_db(None)
        db.close()


def test_register_result_validates_declared_operation_type(tmp_path):
    """register_result rejects a parameter outside the operation type's .dic
    schema (validation is wired at the boundary; nothing is persisted)."""
    from chisurf.core.mfdb.result_registry import register_result

    db = _db(tmp_path)
    try:
        src = _raw(db, tmp_path)
        with pytest.raises(OperationParameterError):
            register_result(
                kind="processed_data",
                data={"x": [0], "y": [0]},
                parent_artifact_id=src,
                operation_type="microtime_shift",
                parameters={"global_shift": 0, "bogus_param": 1},
                db=db,
            )
    finally:
        set_global_db(None)
        db.close()


def test_register_operation_rejects_unknown_parameter(tmp_path):
    db = _db(tmp_path)
    try:
        with pytest.raises(OperationParameterError):
            register_operation(
                operation_type="burst_selection",
                parameters={"min_photons": 60, "bogus": 1},
                db=db,
            )
        # nothing partially recorded (validation is before the write)
        n = db.conn.execute(
            "SELECT COUNT(*) FROM mfdb_operation WHERE operation_type='burst_selection'"
        ).fetchone()[0]
        assert n == 0
    finally:
        set_global_db(None)
        db.close()
