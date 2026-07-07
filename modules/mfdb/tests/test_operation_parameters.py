"""PRD-11: operation-parameter schema — seeding, query, validation."""

from __future__ import annotations

import os

import pytest

from mfdb.provenance.operation_parameters import (
    OperationParameterError,
    get_operation_parameter_defs,
    load_operation_parameter_defs,
    validate_operation_parameters,
)
from mfdb.repository import MFDatabase


def _db(tmp_path) -> MFDatabase:
    return MFDatabase(os.path.join(tmp_path, "ops.db"))


def test_defs_load_from_authored_source():
    defs = load_operation_parameter_defs()
    types = {d.operation_type for d in defs}
    assert {"microtime_shift", "burst_selection"} <= types
    # the repeatable per-channel shift carries a role domain
    shift = next(d for d in defs if d.operation_type == "microtime_shift" and d.name == "shift")
    assert shift.repeatable and shift.role_domain == "detector_channel"


def test_defs_seeded_into_fresh_db(tmp_path):
    db = _db(tmp_path)
    try:
        defs = get_operation_parameter_defs(db.conn, "burst_selection")
        assert "min_photons" in defs
        assert defs["min_photons"].required is True
        assert defs["min_photons"].value_type == "int"
        assert get_operation_parameter_defs(db.conn, "microtime_shift")["shift"].repeatable
    finally:
        db.close()


def test_validation_rejects_unknown_parameter(tmp_path):
    db = _db(tmp_path)
    try:
        with pytest.raises(OperationParameterError):
            validate_operation_parameters(
                db.conn, "burst_selection", {"min_photons": 60, "not_a_param": 1}
            )
    finally:
        db.close()


def test_validation_rejects_missing_required(tmp_path):
    db = _db(tmp_path)
    try:
        with pytest.raises(OperationParameterError):
            validate_operation_parameters(db.conn, "burst_selection", {"time_window": 1.0})
    finally:
        db.close()


def test_validation_accepts_declared_set(tmp_path):
    db = _db(tmp_path)
    try:
        # required min_photons present, others optional and declared
        validate_operation_parameters(
            db.conn, "burst_selection", {"min_photons": 60, "time_window": 1.0}
        )
    finally:
        db.close()


def test_validation_noop_for_undeclared_operation_type(tmp_path):
    db = _db(tmp_path)
    try:
        # No declared schema -> unvalidated (backward compatible), must not raise.
        validate_operation_parameters(db.conn, "some_unknown_op", {"anything": 1})
    finally:
        db.close()
