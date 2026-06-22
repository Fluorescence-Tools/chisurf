"""PRD-16: the Microtime Shifter conforms to the transformer contract."""

from __future__ import annotations

import os

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.transform import check_transformer_conformance, get_transformer
from chisurf.plugins.tttr.tttr_microtime_shifter.api.transformer import (
    MicrotimeShifterTransformer,
    _channel_shifts_from_parameters,
)


def test_microtime_shifter_is_conformant(tmp_path):
    db = MFDatabase(os.path.join(tmp_path, "t.db"))
    try:
        # Declares typed ports AND its operation_type has a .dic parameter schema.
        check_transformer_conformance(MicrotimeShifterTransformer(), conn=db.conn)
    finally:
        db.close()


def test_microtime_shifter_self_registers():
    assert get_transformer("microtime_shifter") is not None


def test_role_indexed_shift_maps_to_channel_shifts():
    # register_operation shape: list of {value, role=channel}
    params = {"global_shift": 2, "shift": [{"value": 5, "role": "0"}, {"value": 3, "role": "1"}]}
    assert _channel_shifts_from_parameters(params) == {0: 5, 1: 3}
    # plain mapping is also accepted
    assert _channel_shifts_from_parameters({"shift": {2: 7}}) == {2: 7}


def test_ports_declare_expected_kinds():
    t = MicrotimeShifterTransformer()
    assert t.input_spec[0].accepts_kind("raw_measurement")
    assert t.input_spec[0].accepts_format("ptu")
    assert t.output_spec[0].accepts_kind("processed_data")
