"""Replay executor for the ``microtime_shift`` operation (PRD-21 Task 2 follow-on).

PRD-21 Task 2 records each derived artifact's producing operation as a replayable
:class:`~chisurf.core.mfdb.provenance.compute_spec.ComputeSpec`; ``recompute``/``replay`` dispatch
to a registered executor per ``operation_type``. This module fills that seam for the
Micro-time Shifter: it materializes each source artifact's stored TTTR file, re-runs
the conformant transformer with the spec's parameters, and registers each shifted
output as a new ``processed_data`` artifact derived from its source.

Importing this module self-registers the executor (the same idiom as
``api/transformer.py``), so ``from ...api import replay`` is all a caller needs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from chisurf.core.mfdb.provenance.compute_spec import ComputeSpec, register_replay_executor
from chisurf.core.mfdb.provenance.result_registry import register_result
from chisurf.core.transform import TransformInputs
from chisurf.plugins.tttr.tttr_microtime_shifter.api.transformer import (
    MICROTIME_SHIFTER,
    OPERATION_TYPE,
)


def microtime_shift_replay_executor(spec: ComputeSpec, db: Any) -> str:
    """Re-run a ``microtime_shift`` from its compute spec; return the new artifact id.

    Materializes each source artifact, re-runs the transformer into a fresh temp
    directory with the spec's parameters (untouched — the ``output_dir`` is added only
    on the transform call, not on the recorded operation), and registers each shifted
    output as a new ``processed_data`` artifact derived from its source. Returns the
    first new artifact id (microtime_shift normally has a single source/output).
    """
    if not spec.source_artifact_ids:
        raise ValueError("microtime_shift replay needs at least one source artifact")

    work_dir = tempfile.mkdtemp(prefix="mfdb_replay_shift_")
    paths = [
        db.materialize_artifact_file(aid, into=work_dir)
        for aid in spec.source_artifact_ids
    ]
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    transform_params = dict(spec.parameters)
    transform_params["output_dir"] = out_dir

    result = MICROTIME_SHIFTER.transform(
        TransformInputs(files=tuple(paths)), transform_params
    )
    shifted = result.outputs.get("shifted", {})

    new_ids: list[str] = []
    for source_id, in_path in zip(spec.source_artifact_ids, paths):
        out_path = shifted.get(in_path)
        if not out_path:
            continue
        new_id = register_result(
            kind="processed_data",
            data=out_path,
            parent_artifact_id=source_id,
            operation_type=spec.operation_type,
            parameters=spec.parameters,
            metadata={"plugin": "microtime_shifter", "replay": True},
            data_format=Path(out_path).suffix.lstrip(".") or "tttr",
            db=db,
        )
        if new_id:
            new_ids.append(new_id)

    if not new_ids:
        raise RuntimeError("microtime_shift replay produced no output artifact")
    return new_ids[0]


#: Self-register on import (idempotent), mirroring ``api/transformer.py``.
register_replay_executor(OPERATION_TYPE, microtime_shift_replay_executor)
