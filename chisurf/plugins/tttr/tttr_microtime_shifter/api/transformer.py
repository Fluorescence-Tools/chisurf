"""Microtime Shifter as a conformant transformer (PRD-16 / PRD-11).

Adapts the plugin's pure ``shift_file`` to the general transformer contract:
declared typed ports, an ``operation_type`` whose parameter schema lives in the
`.dic` (``mfdb_operation_parameter_def``: ``global_shift`` + the repeatable
role-indexed ``shift``), and a pure ``transform``. No Qt, no DB.
"""

from __future__ import annotations

from typing import Any

from chisurf.core.transform import PortSpec, TransformInputs, TransformResult, register_transformer
from chisurf.plugins.tttr.tttr_microtime_shifter.api.shift import shift_file

#: Vocabulary value shared with mfdb_operation_parameter_def / register_operation.
OPERATION_TYPE = "microtime_shift"


def _channel_shifts_from_parameters(parameters: dict) -> dict[int, int]:
    """Build ``{channel: shift}`` from the role-indexed ``shift`` parameter.

    Accepts the register_operation shape (a list of ``{"value", "role"}`` entries,
    role = detector channel) or a plain ``{channel: shift}`` mapping.
    """
    shift = parameters.get("shift")
    if isinstance(shift, dict):
        return {int(k): int(v) for k, v in shift.items()}
    out: dict[int, int] = {}
    for entry in shift or []:
        if isinstance(entry, dict) and entry.get("role") is not None:
            out[int(entry["role"])] = int(entry.get("value", 0))
    return out


class MicrotimeShifterTransformer:
    """Conformant transformer over the ``microtime_shift`` operation type."""

    transformer_id = "microtime_shifter"
    operation_type = OPERATION_TYPE
    version = "1.0"
    input_spec = [
        PortSpec(
            name="raw",
            kinds=("raw_measurement",),
            formats=("ptu", "spc", "ht3", "hdf", "h5"),
        )
    ]
    output_spec = [PortSpec(name="shifted", kinds=("processed_data",))]

    def transform(self, inputs: TransformInputs, parameters: dict) -> TransformResult:
        """Pure transform: shift each input TTTR file. No Qt, no DB."""
        global_shift = int(parameters.get("global_shift", 0) or 0)
        channel_shifts = _channel_shifts_from_parameters(parameters)
        filetype = parameters.get("filetype")
        output_dir = parameters.get("output_dir")
        outputs: dict[str, Any] = {}
        applied: dict[str, Any] = {}
        for path in inputs.files:
            out_path, applied_shifts = shift_file(
                path,
                global_shift=global_shift,
                channel_shifts=channel_shifts,
                filetype=filetype,
                output_dir=output_dir,
            )
            outputs[path] = out_path
            applied[path] = applied_shifts
        return TransformResult(outputs={"shifted": outputs}, warnings=[])


#: Registered instance for discovery / conformance enumeration.
MICROTIME_SHIFTER = register_transformer(MicrotimeShifterTransformer())
