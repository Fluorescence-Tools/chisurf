"""Replay executor for the ``burst_selection`` operation (PRD-21 Task 2 follow-on).

Mirrors ``tttr_microtime_shifter/api/replay.py`` for burst selection: it materializes
the source artifact's stored TTTR file, reconstructs an :class:`AnalysisRequest` from the
compute spec's parameters (``settings_from_parameters`` is the inverse of
``extract_burst_parameters``), re-runs the pure ``analyze_request``, and registers the
outputs through :class:`BurstMFDBPipeline` so the replay lands in the same shape as a
normal run (PRD-28: a burst-table artifact derived from the source plus its sidecars).

Faithful replay depends on the compute spec capturing the full reproducible parameter
set — including the detector ``channels`` stream mask (role-indexed) and the GMM
determinism fields — which the ``burst_selection`` ``.dic`` schema now declares.

Importing this module self-registers the executor (the same idiom as ``api/transformer``).
"""

from __future__ import annotations

import tempfile
from typing import Any

from chisurf.core.mfdb.provenance.compute_spec import ComputeSpec, register_replay_executor
from chisurf.plugins.burst.burst_selection.api.mfdb import BurstMFDBPipeline
from chisurf.plugins.burst.burst_selection.api.models import AnalysisRequest, MFDBContext
from chisurf.plugins.burst.burst_selection.api.selection import analyze_request
from chisurf.plugins.burst.burst_selection.api.transformer import (
    OPERATION_TYPE,
    settings_from_parameters,
)


def burst_selection_replay_executor(spec: ComputeSpec, db: Any) -> str:
    """Re-run a ``burst_selection`` from its compute spec; return the new artifact id.

    Returns the new burst-table artifact id (burst selection has a single TTTR
    source). Raises if the run produced no burst table.
    """
    if not spec.source_artifact_ids:
        raise ValueError("burst_selection replay needs a source artifact")
    source_id = spec.source_artifact_ids[0]

    work_dir = tempfile.mkdtemp(prefix="mfdb_replay_burst_")
    src_path = db.materialize_artifact_file(source_id, into=work_dir)

    request = AnalysisRequest(
        files=[src_path],
        settings=settings_from_parameters(spec.parameters),
        filetype=spec.parameters.get("filetype"),
        output_dir=work_dir,
        mfdb=MFDBContext(
            enabled=True,
            register_missing_inputs=False,
            # Link the replayed burst table to the original source artifact rather
            # than re-registering the materialized temp copy as a new raw input.
            source_artifact_ids={src_path: source_id},
        ),
    )
    result = analyze_request(request)
    registration = BurstMFDBPipeline(db=db).register_run(request, result)

    if not registration.burst_table_artifacts:
        raise RuntimeError(
            "burst_selection replay produced no burst table"
            + (f": {registration.warnings}" if registration.warnings else "")
        )
    return next(iter(registration.burst_table_artifacts.values()))


#: Self-register on import (idempotent), mirroring ``api/transformer.py``.
register_replay_executor(OPERATION_TYPE, burst_selection_replay_executor)
