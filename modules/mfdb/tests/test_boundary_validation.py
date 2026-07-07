"""PRD-26 Task 4: dictionary-driven boundary parameter validation.

Proves the boundary validator is derived from the ``.dic`` (unknown columns,
missing mandatory columns, type-incoherent values, and out-of-vocabulary values are
rejected; declared payloads pass) and that the PRD-11 operation-parameter validation
now also enforces value type and numeric bounds — sharing one type check.
"""

from __future__ import annotations

import os

import pytest

from mfdb.security.boundary_validation import (
    BoundaryValidationError,
    DictionaryValidator,
    check_value_type,
)
from mfdb.provenance.operation_parameters import (
    OperationParameterError,
    validate_operation_parameters,
)
from mfdb.repository import MFDatabase


@pytest.fixture(scope="module")
def validator() -> DictionaryValidator:
    return DictionaryValidator.load_bundled()


# --- check_value_type (shared kernel) ---------------------------------------


def test_check_value_type_accepts_coercible_and_rejects_incoherent():
    assert check_value_type(5, "int") is None
    assert check_value_type("5", "int") is None  # stringly-typed payloads pass
    assert check_value_type("x", "int") is not None
    assert check_value_type(True, "int") is not None  # bool is not an int here
    assert check_value_type(1.0, "int") is None  # SQLite REAL round-trip of an int
    assert check_value_type(1.5, "int") is not None  # genuine fraction rejected
    assert check_value_type(1.5, "float") is None
    assert check_value_type("1.5", "float") is None
    assert check_value_type("nan-ish", "float") is not None
    assert check_value_type(True, "boolean") is None
    assert check_value_type("yes", "boolean") is None
    assert check_value_type(2, "boolean") is not None
    assert check_value_type(None, "int") is None  # nullability handled elsewhere
    assert check_value_type("anything", "text") is None  # unknown kind: free


def test_check_value_type_positive_int_rejects_negative():
    assert check_value_type(0, "positive_int") is None
    assert check_value_type(-1, "positive_int") is not None


# --- DictionaryValidator (table boundary) -----------------------------------


def test_validator_rejects_unknown_column(validator):
    with pytest.raises(BoundaryValidationError, match="Unknown column"):
        validator.validate(
            "flr_sample",
            {
                "sample_id": 1,
                "num_of_probes": 1,
                "sample_condition_id": 1,
                "entity_assembly_id": 1,
                "solvent_phase": "liquid",
                "not_a_real_column": 1,
            },
        )


def test_validator_rejects_missing_mandatory(validator):
    with pytest.raises(BoundaryValidationError, match="Missing required column"):
        validator.validate("flr_sample", {"sample_id": 1})


def test_validator_partial_skips_mandatory(validator):
    # partial update: mandatory columns not required, declared values still checked
    validator.validate(
        "flr_sample", {"description": "edit"}, require_mandatory=False
    )


def test_validator_rejects_type_incoherent_value(validator):
    with pytest.raises(BoundaryValidationError, match="expected integer"):
        validator.validate(
            "flr_sample",
            {
                "sample_id": "not-an-int",
                "num_of_probes": 1,
                "sample_condition_id": 1,
                "entity_assembly_id": 1,
                "solvent_phase": "liquid",
            },
        )


def test_validator_rejects_out_of_vocabulary_value(validator):
    with pytest.raises(BoundaryValidationError, match="not in allowed values"):
        validator.validate(
            "flr_sample",
            {
                "sample_id": 1,
                "num_of_probes": 1,
                "sample_condition_id": 1,
                "entity_assembly_id": 1,
                "solvent_phase": "plasma",  # not a declared solvent_phase
            },
        )


def test_validator_accepts_declared_payload(validator):
    validator.validate(
        "flr_sample",
        {
            "sample_id": 1,
            "num_of_probes": 2,
            "sample_condition_id": 1,
            "entity_assembly_id": 1,
            "solvent_phase": "liquid",
            "sample_type": "protein",
            "description": "ok",
        },
    )


def test_validator_unknown_table_is_noop(validator):
    validator.validate("not_a_table", {"whatever": 1})


def test_validator_allow_unknown_bypasses_column_whitelist(validator):
    validator.validate(
        "flr_sample",
        {"description": "x", "extra": 1},
        require_mandatory=False,
        allow_unknown=True,
    )


# --- operation-parameter validation now type/bound aware --------------------


def _db(tmp_path) -> MFDatabase:
    return MFDatabase(os.path.join(tmp_path, "ops.db"))


def test_operation_parameters_reject_out_of_bounds(tmp_path):
    db = _db(tmp_path)
    try:
        with pytest.raises(OperationParameterError, match="below lower bound"):
            # min_photons declares lower_bound=1
            validate_operation_parameters(
                db.conn, "burst_selection", {"min_photons": 0}
            )
    finally:
        db.close()


def test_operation_parameters_reject_wrong_type(tmp_path):
    db = _db(tmp_path)
    try:
        with pytest.raises(OperationParameterError, match="expected integer"):
            validate_operation_parameters(
                db.conn, "burst_selection", {"min_photons": "lots"}
            )
    finally:
        db.close()


def test_operation_parameters_accept_valid_and_rich_and_repeatable(tmp_path):
    db = _db(tmp_path)
    try:
        # scalar within bounds
        validate_operation_parameters(
            db.conn, "burst_selection", {"min_photons": 60}
        )
        # repeatable, role-indexed; rich-dict inner value type-checked
        validate_operation_parameters(
            db.conn,
            "microtime_shift",
            {
                "global_shift": {"value": 5, "units": "ns"},
                "shift": [{"value": 5, "role": "0"}, {"value": 3, "role": "1"}],
            },
        )
    finally:
        db.close()
