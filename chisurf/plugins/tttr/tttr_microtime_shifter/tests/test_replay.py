"""PRD-21 Task 2 follow-on: the microtime_shift replay executor.

Registers a real TTTR file as a raw measurement, records a microtime_shift result,
then recomputes/replays that artifact's compute spec end to end — materializing the
source from the object store, re-running the conformant transformer, and registering a
new derived artifact.
"""

from __future__ import annotations

import os

import pytest

tttrlib = pytest.importorskip("tttrlib")

from chisurf.core.mfdb.provenance.compute_spec import (
    NoReplayExecutorError,
    get_compute_spec,
    recompute,
    replay,
    unregister_replay_executor,
)
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.provenance.result_registry import (
    register_raw_measurement,
    register_result,
    set_global_db,
)

# Importing the plugin module self-registers the replay executor.
from chisurf.plugins.tttr.tttr_microtime_shifter.api import replay as _replay  # noqa: F401
from chisurf.plugins.tttr.tttr_microtime_shifter.api.transformer import OPERATION_TYPE

_FIXTURES = (
    "test/data/clsm/Leica_SP5.ptu",
    "test/data/clsm/Leica_SP8.ptu",
    "test/data/test_µ_file.ptu",
)


def _fixture_path() -> str:
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    for rel in _FIXTURES:
        candidate = os.path.join(root, rel)
        if os.path.isfile(candidate):
            return candidate
    pytest.skip("no TTTR PTU fixture available for replay test")


@pytest.fixture
def chain(tmp_path):
    """Raw artifact (real PTU in the object store) + a microtime_shift result."""
    src = _fixture_path()
    # Re-register idempotently: the replay-executor registry is process-global and
    # other test files exercise/clear the "microtime_shift" slot, so don't rely on
    # import-time registration surviving cross-file ordering.
    _replay.register_replay_executor(
        OPERATION_TYPE, _replay.microtime_shift_replay_executor
    )
    db = MFDatabase(os.path.join(tmp_path, "replay.db"))
    raw = register_raw_measurement(src, db=db)
    shifted = register_result(
        kind="processed_data",
        data=src,  # any payload; the spec/source is what replay reads
        parent_artifact_id=raw,
        operation_type=OPERATION_TYPE,
        parameters={"global_shift": 3, "shift": [{"value": 1, "role": "0"}]},
        db=db,
    )
    try:
        yield db, {"raw": raw, "shifted": shifted}
    finally:
        set_global_db(None)
        db.close()


def test_recompute_reruns_pipeline_and_registers_new_artifact(chain):
    db, ids = chain
    new_id = recompute(db, ids["shifted"])
    assert new_id and new_id not in (ids["raw"], ids["shifted"])
    # the new artifact is processed_data derived from the same raw source
    art = db.get_artifact(new_id)
    assert art["artifact_kind"] == "processed_data"
    assert db.get_artifact_ancestors(new_id) == [ids["raw"]]
    # its recorded spec carries the replayed parameters
    spec = get_compute_spec(db, new_id)
    assert spec.operation_type == OPERATION_TYPE
    assert spec.parameters["global_shift"] == 3


def test_replay_with_override_applies_new_parameter(chain):
    db, ids = chain
    new_id = replay(db, ids["shifted"], {"global_shift": 9})
    spec = get_compute_spec(db, new_id)
    assert spec.parameters["global_shift"] == 9
    assert db.get_artifact_ancestors(new_id) == [ids["raw"]]


def test_executor_is_registered_for_operation_type(chain):
    from chisurf.core.mfdb.provenance.compute_spec import get_replay_executor

    assert get_replay_executor(OPERATION_TYPE) is not None


def test_missing_executor_raises_after_unregister(chain):
    db, ids = chain
    unregister_replay_executor(OPERATION_TYPE)
    try:
        with pytest.raises(NoReplayExecutorError):
            recompute(db, ids["shifted"])
    finally:
        # restore for any later-collected tests sharing the process registry
        _replay.register_replay_executor(
            OPERATION_TYPE, _replay.microtime_shift_replay_executor
        )
