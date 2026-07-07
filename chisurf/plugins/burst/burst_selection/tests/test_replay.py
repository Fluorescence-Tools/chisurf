"""PRD-21 Task 2 follow-on: the burst_selection replay executor.

Registers a real single-molecule SPC file as a raw measurement, records a
burst_selection result whose compute spec carries the full reproducible parameter set
(including the role-indexed detector ``channels`` mask and GMM determinism), then
recomputes that artifact end to end — materializing the source from the object store,
re-running the real burst pipeline, and registering a new burst-table artifact.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("tttrlib")

from chisurf.core.mfdb.provenance.compute_spec import get_compute_spec, recompute
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.provenance.result_registry import (
    register_raw_measurement,
    register_result,
    set_global_db,
)
from chisurf.plugins.burst.burst_selection.api import replay as _replay  # noqa: F401
from chisurf.plugins.burst.burst_selection.api.mfdb import extract_burst_parameters
from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisRequest,
    AnalysisSettings,
    BurstDetectionSettings,
    DeltaMacroTimeFilterSettings,
    GMMSettings,
    PhotonFilterSettings,
)
from chisurf.plugins.burst.burst_selection.api.transformer import OPERATION_TYPE

_SPC = (
    Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna" / "m000.spc"
)
_STREAM_CHANNELS = [0, 1, 8, 9]


def _settings() -> AnalysisSettings:
    s = AnalysisSettings()
    s.photon_filter = PhotonFilterSettings(
        channels=list(_STREAM_CHANNELS),
        filter_active=False,
        delta_macro_time_filter=DeltaMacroTimeFilterSettings(dT_min=0.0),
    )
    s.burst_detection = BurstDetectionSettings(
        min_photons=20, photon_window=10, time_window=1e-3
    )
    s.gmm = GMMSettings(
        covariance_type="spherical", random_state=42, max_iter=50, n_init=1
    )
    return s


@pytest.fixture
def chain(tmp_path):
    """Raw SPC artifact + a burst_selection result carrying reproducible params."""
    if not _SPC.is_file():
        pytest.skip("bundled SPC fixture not available")
    _replay.register_replay_executor(
        OPERATION_TYPE, _replay.burst_selection_replay_executor
    )
    db = MFDatabase(os.path.join(tmp_path, "burst_replay.db"))
    raw = register_raw_measurement(str(_SPC), db=db)
    params = extract_burst_parameters(AnalysisRequest(files=[], settings=_settings()))
    # Metadata-only setup artifact: replay reads the operation/params/source from the
    # compute spec, not the stored payload, so we don't need a full BurstTable here.
    burst = register_result(
        kind="burst_table",
        data=None,
        parent_artifact_id=raw,
        operation_type=OPERATION_TYPE,
        parameters=params,
        db=db,
    )
    try:
        yield db, {"raw": raw, "burst": burst}, params
    finally:
        set_global_db(None)
        db.close()


def test_burst_params_capture_channels_and_gmm_determinism(chain):
    db, ids, params = chain
    # the reproducible set the schema now declares
    assert {(e["value"], e["role"]) for e in params["channels"]} == {
        (0, "0"), (1, "1"), (8, "2"), (9, "3")
    }
    assert params["gmm_covariance_type"] == "spherical"
    assert params["gmm_random_state"] == 42
    # round-trips through the recorded compute spec
    spec = get_compute_spec(db, ids["burst"])
    assert sorted(int(e["value"]) for e in spec.parameters["channels"]) == _STREAM_CHANNELS
    assert spec.parameters["gmm_random_state"] == 42


def test_recompute_reruns_burst_pipeline_and_registers_table(chain):
    db, ids, _ = chain
    new_id = recompute(db, ids["burst"])
    assert new_id and new_id not in (ids["raw"], ids["burst"])
    art = db.get_artifact(new_id)
    assert art["artifact_kind"] == "burst_table"
    # the replayed table is derived from the original raw source (not the temp copy)
    assert db.get_artifact_ancestors(new_id) == [ids["raw"]]
    # and its own spec preserves the reproducible channel mask
    spec = get_compute_spec(db, new_id)
    assert sorted(int(e["value"]) for e in spec.parameters["channels"]) == _STREAM_CHANNELS
