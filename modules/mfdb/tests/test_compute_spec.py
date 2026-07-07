"""PRD-21 Task 2: embedded replayable compute specs.

A derived artifact's producing operation (operation_type + role-indexed parameters +
source artifact ids) is read back as a replayable unit; ``with_overrides`` is the
"what-if"; ``recompute``/``replay`` dispatch to a registered executor.
"""

from __future__ import annotations

import os

import pytest

from mfdb.provenance.compute_spec import (
    ComputeSpec,
    NoReplayExecutorError,
    get_compute_spec,
    get_replay_executor,
    recompute,
    register_replay_executor,
    replay,
    unregister_replay_executor,
)


def _restore_executor(operation_type, prior):
    """Restore a previously-registered executor (the registry is process-global)."""
    if prior is None:
        unregister_replay_executor(operation_type)
    else:
        register_replay_executor(operation_type, prior)
from mfdb.repository import MFDatabase
from mfdb.provenance.result_registry import (
    register_operation,
    register_raw_measurement,
    register_result,
    set_global_db,
)


@pytest.fixture
def chain(tmp_path):
    db = MFDatabase(os.path.join(tmp_path, "cs.db"))
    f = tmp_path / "m.ptu"
    f.write_bytes(b"\x00\x01\x02")
    raw = register_raw_measurement(str(f), db=db)
    shifted = register_result(
        kind="processed_data",
        data={"x": [0], "y": [1]},
        parent_artifact_id=raw,
        operation_type="microtime_shift",
        parameters={"global_shift": 5},
        db=db,
    )
    try:
        yield db, {"raw": raw, "shifted": shifted}
    finally:
        set_global_db(None)
        db.close()


def test_get_compute_spec_reads_operation_params_and_sources(chain):
    db, ids = chain
    spec = get_compute_spec(db, ids["shifted"])
    assert spec is not None
    assert spec.operation_type == "microtime_shift"
    assert spec.source_artifact_ids == (ids["raw"],)
    assert spec.parameters["global_shift"] == 5
    assert spec.operation_id  # the recorded producing operation


def test_role_indexed_parameters_round_trip(tmp_path):
    db = MFDatabase(os.path.join(tmp_path, "cs2.db"))
    try:
        f = tmp_path / "m.ptu"
        f.write_bytes(b"\x00\x01")
        raw = register_raw_measurement(str(f), db=db)
        op = register_operation(
            operation_type="microtime_shift",
            inputs=[raw],
            outputs=[],
            parameters={
                "global_shift": 0,
                "shift": [{"value": 3, "role": "0"}, {"value": 7, "role": "1"}],
            },
            db=db,
        )
        # the operation's own output is empty here; read the spec via the operation's
        # recorded parameters by registering an output artifact and re-reading
        out = register_result(
            kind="processed_data",
            data={"x": [0], "y": [1]},
            parent_artifact_id=raw,
            operation_type="microtime_shift",
            parameters={
                "global_shift": 0,
                "shift": [{"value": 3, "role": "0"}, {"value": 7, "role": "1"}],
            },
            db=db,
        )
        spec = get_compute_spec(db, out)
        assert spec.parameters["global_shift"] == 0
        # role-indexed param reconstructed as a list of {value, role}
        shift = spec.parameters["shift"]
        assert isinstance(shift, list)
        assert {(e["value"], e["role"]) for e in shift} == {(3.0, "0"), (7.0, "1")}
    finally:
        set_global_db(None)
        db.close()


def test_with_overrides_is_a_whatif_without_operation_id(chain):
    db, ids = chain
    spec = get_compute_spec(db, ids["shifted"])
    whatif = spec.with_overrides({"global_shift": 9})
    assert whatif.parameters["global_shift"] == 9
    assert whatif.operation_type == spec.operation_type
    assert whatif.source_artifact_ids == spec.source_artifact_ids
    assert whatif.operation_id == ""  # hypothetical, not recorded
    assert spec.parameters["global_shift"] == 5  # original unchanged (frozen)


def test_root_artifact_with_no_producing_op_has_no_spec(chain):
    db, ids = chain
    # insert a bare artifact with no operation link -> no compute spec
    db.dao.insert(
        "mfdb_artifact",
        {"artifact_id": "bare", "artifact_kind": "raw_measurement", "storage_mode": "embedded"},
    )
    assert get_compute_spec(db, "bare") is None


def test_recompute_and_replay_dispatch_to_registered_executor(chain):
    db, ids = chain
    calls = []

    def fake_executor(spec: ComputeSpec, database) -> str:
        calls.append(spec)
        return "new-artifact"

    prior = get_replay_executor("microtime_shift")
    register_replay_executor("microtime_shift", fake_executor)
    try:
        assert recompute(db, ids["shifted"]) == "new-artifact"
        assert calls[-1].parameters["global_shift"] == 5
        # replay applies the override before executing
        assert replay(db, ids["shifted"], {"global_shift": 42}) == "new-artifact"
        assert calls[-1].parameters["global_shift"] == 42
    finally:
        _restore_executor("microtime_shift", prior)


def test_recompute_without_executor_raises(chain):
    db, ids = chain
    # the plugin executor may be registered process-globally; remove it for this case
    prior = get_replay_executor("microtime_shift")
    unregister_replay_executor("microtime_shift")
    try:
        with pytest.raises(NoReplayExecutorError):
            recompute(db, ids["shifted"])
    finally:
        _restore_executor("microtime_shift", prior)
