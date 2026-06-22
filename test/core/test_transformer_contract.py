"""PRD-16: the general transformer contract — registry + conformance check."""

from __future__ import annotations

import os

import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.transform import (
    PortSpec,
    TransformInputs,
    TransformResult,
    TransformerConformanceError,
    check_transformer_conformance,
    list_transformers,
    register_transformer,
)


class _ShiftTransformer:
    """A minimal conformant transformer over the declared microtime_shift schema."""

    transformer_id = "microtime_shifter_demo"
    operation_type = "microtime_shift"
    version = "1.0"
    input_spec = [PortSpec(name="raw", kinds=("raw_measurement",), formats=("ptu", "spc"))]
    output_spec = [PortSpec(name="shifted", kinds=("processed_data",))]

    def transform(self, inputs: TransformInputs, parameters: dict) -> TransformResult:
        return TransformResult(outputs={"shifted": inputs.files})


def test_conformant_transformer_passes(tmp_path):
    db = MFDatabase(os.path.join(tmp_path, "t.db"))
    try:
        # microtime_shift has a .dic parameter schema (seeded), so the DB-aware
        # conformance check passes too.
        check_transformer_conformance(_ShiftTransformer(), conn=db.conn)
    finally:
        db.close()


def test_missing_ports_fails():
    class _NoPorts:
        transformer_id = "bad"
        operation_type = "microtime_shift"
        version = "1.0"
        input_spec = []
        output_spec = [PortSpec(name="o", kinds=("processed_data",))]

        def transform(self, inputs, parameters):
            return TransformResult()

    with pytest.raises(TransformerConformanceError):
        check_transformer_conformance(_NoPorts())


def test_operation_type_without_schema_fails(tmp_path):
    class _NoSchema:
        transformer_id = "noschema"
        operation_type = "totally_undeclared_op"
        version = "1.0"
        input_spec = [PortSpec(name="i", kinds=("raw_measurement",))]
        output_spec = [PortSpec(name="o", kinds=("processed_data",))]

        def transform(self, inputs, parameters):
            return TransformResult()

    db = MFDatabase(os.path.join(tmp_path, "t.db"))
    try:
        with pytest.raises(TransformerConformanceError):
            check_transformer_conformance(_NoSchema(), conn=db.conn)
    finally:
        db.close()


def test_registry_enumerates():
    t = _ShiftTransformer()
    register_transformer(t)
    assert any(x.transformer_id == "microtime_shifter_demo" for x in list_transformers())


def test_port_spec_kind_and_format_matching():
    spec = PortSpec(name="raw", kinds=("raw_measurement",), formats=("ptu",))
    assert spec.accepts_kind("raw_measurement")
    assert not spec.accepts_kind("processed_data")
    assert spec.accepts_format("ptu")
    assert not spec.accepts_format("hdf")
    assert PortSpec(name="x", kinds=("k",)).accepts_format("anything")  # no format constraint
