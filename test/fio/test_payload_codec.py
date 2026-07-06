"""Tests for MFDB payload msgpack codecs."""
from __future__ import annotations

import msgpack
import numpy as np
import pytest

from chisurf.core.mfdb.payload_codec import (
    MIGRATIONS,
    REGISTRY,
    PayloadSchemaError,
    decode_payload,
    encode_payload,
    get_payload_schema,
)
from chisurf.core.mfdb.payload_models import (
    AnisotropyCurve,
    BurstSelection,
    BurstTable,
    FcsCorrelation,
    GenericCurve,
    PdaHistogram,
    Spectrum,
    TcspcDecay,
    TttrPhotonStream,
    TttrReference,
)


def _roundtrip(kind: str, payload):
    blob, data_format = encode_payload(kind, payload)
    assert data_format == "msgpack"
    return decode_payload(blob)


def test_fcs_correlation_roundtrip_preserves_dtype_shape_and_nan_inf():
    lag = np.array([1e-6, 2e-6, 3e-6], dtype=np.float64)
    correlation = np.array([[1.0, np.nan, np.inf], [0.9, 0.8, -np.inf]], dtype=np.float64)
    error = np.full(correlation.shape, 0.1, dtype=np.float64)
    payload = FcsCorrelation(
        lag=lag,
        correlation=correlation,
        error=error,
        curve_names=["ACF_g", "CCF"],
    )

    decoded = _roundtrip("fcs_correlation", payload)

    np.testing.assert_array_equal(decoded.lag, lag)
    np.testing.assert_array_equal(decoded.correlation, correlation)
    np.testing.assert_array_equal(decoded.error, error)
    assert decoded.correlation.dtype == np.dtype("float64")
    assert decoded.curve_names == ["ACF_g", "CCF"]
    assert get_payload_schema("fcs_correlation")["units"]["lag"] == "s"


def test_spectrum_roundtrip():
    payload = Spectrum(
        wavelength=np.array([500.0, 510.0], dtype=np.float64),
        intensity=np.array([0.2, 1.0], dtype=np.float64),
        spectrum_type="emission",
        normalized=True,
    )

    decoded = _roundtrip("spectra", payload)

    np.testing.assert_array_equal(decoded.wavelength, payload.wavelength)
    np.testing.assert_array_equal(decoded.intensity, payload.intensity)
    assert decoded.spectrum_type == "emission"
    assert decoded.normalized is True


def test_tcspc_decay_roundtrip_with_microtime_resolution():
    payload = TcspcDecay(
        time=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        counts=np.array([10, 20, 5], dtype=np.int64),
        irf=np.array([1.0, 0.5, 0.1], dtype=np.float64),
        channel="green",
        adc_resolution_ns=0.1,
        micro_time_resolution_ns=0.004,
    )

    decoded = _roundtrip("tcspc_decay", payload)

    np.testing.assert_array_equal(decoded.counts, payload.counts)
    assert decoded.counts.dtype == np.dtype("int64")
    assert decoded.micro_time_resolution_ns == 0.004


def test_anisotropy_curve_roundtrip_with_l1_l2():
    payload = AnisotropyCurve(
        time=np.array([0.0, 1.0], dtype=np.float64),
        vv=np.array([100.0, 80.0], dtype=np.float64),
        vh=np.array([40.0, 30.0], dtype=np.float64),
        l1=np.array([1.0, 0.8], dtype=np.float64),
        l2=np.array([0.2, 0.1], dtype=np.float64),
        g_factor=1.03,
    )

    decoded = _roundtrip("anisotropy_curve", payload)

    np.testing.assert_array_equal(decoded.l1, payload.l1)
    np.testing.assert_array_equal(decoded.l2, payload.l2)
    assert decoded.g_factor == 1.03


def test_payload_schema_exposes_flrcif_item_ids_for_matched_fields():
    aniso_schema = get_payload_schema("anisotropy_curve")
    assert aniso_schema["fields"]["l1"]["flrcif_item_id"] == "_flr_chisurf_parameter.l1"
    assert aniso_schema["fields"]["l2"]["flrcif_item_id"] == "_flr_chisurf_parameter.l2"
    assert aniso_schema["fields"]["g_factor"]["flrcif_item_id"] == "_flr_chisurf_parameter.g"

    tcspc_schema = get_payload_schema("tcspc_decay")
    assert tcspc_schema["fields"]["adc_resolution_ns"]["flrcif_item_id"] == "_flr_chisurf_parameter.dtTAC[ns]"
    assert tcspc_schema["fields"]["micro_time_resolution_ns"]["flrcif_item_id"] == "_flr_chisurf_parameter.dtMT[ns]"


def test_payload_flrcif_item_ids_exist_in_bundled_dictionary():
    from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary

    dictionary = MmcifDictionary.load_bundled()
    for kind in REGISTRY:
        schema = get_payload_schema(kind)
        for field in schema["fields"].values():
            flrcif_item_id = field.get("flrcif_item_id")
            if flrcif_item_id:
                assert dictionary.get_item(flrcif_item_id) is not None


def test_payload_flrcif_item_ids_use_existing_parameter_translation():
    from chisurf.core.mfdb.chinet_adapter import _lookup_flrcif_name

    aniso_schema = get_payload_schema("anisotropy_curve")
    assert aniso_schema["fields"]["l1"]["flrcif_item_id"] == _lookup_flrcif_name("l1")
    assert aniso_schema["fields"]["l2"]["flrcif_item_id"] == _lookup_flrcif_name("l2")
    assert aniso_schema["fields"]["g_factor"]["flrcif_item_id"] == _lookup_flrcif_name("g")

    tcspc_schema = get_payload_schema("tcspc_decay")
    assert tcspc_schema["fields"]["adc_resolution_ns"]["flrcif_item_id"] == _lookup_flrcif_name("dtTAC[ns]")
    assert tcspc_schema["fields"]["micro_time_resolution_ns"]["flrcif_item_id"] == _lookup_flrcif_name("dtMT[ns]")


def test_pda_histogram_2d_roundtrip():
    payload = PdaHistogram(
        edges=[
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        ],
        counts=np.ones((2, 3), dtype=np.float64),
        axis_names=["E", "S"],
    )

    decoded = _roundtrip("pda_histogram", payload)

    assert isinstance(decoded.edges, list)
    np.testing.assert_array_equal(decoded.edges[0], payload.edges[0])
    np.testing.assert_array_equal(decoded.counts, payload.counts)
    assert decoded.counts.shape == (2, 3)


def test_burst_table_dataframe_roundtrip_preserves_order_and_dtypes():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "burst_id": np.array([1, 2, 3], dtype=np.int64),
            "duration": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        }
    )

    payload = BurstTable.from_dataframe(df)
    decoded = _roundtrip("burst_table", payload)
    restored = decoded.to_dataframe()

    assert decoded.columns == ["burst_id", "duration"]
    assert decoded.dtypes == ["int64", "float64"]
    assert list(restored.columns) == ["burst_id", "duration"]
    assert str(restored["burst_id"].dtype) == "int64"
    assert str(restored["duration"].dtype) == "float64"


def test_burst_table_dataframe_roundtrip_preserves_string_columns():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "burst_id": np.array([1, 2], dtype=np.int64),
            "label": ["keep", "drop"],
        }
    )

    payload = BurstTable.from_dataframe(df)
    decoded = _roundtrip("burst_table", payload)
    restored = decoded.to_dataframe()

    assert decoded.dtypes[decoded.columns.index("label")].startswith("<U")
    assert restored["label"].tolist() == ["keep", "drop"]


def test_burst_table_string_column_with_missing_values_roundtrips():
    """A string column with missing values (None/NaN) — common in burst summary
    tables, e.g. 'First File' — serializes as a string column, missing -> ''."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "Number of Photons": np.array([100, 200, 300], dtype=np.int64),
            "First File": ["a.ptu", None, "c.ptu"],
        }
    )

    payload = BurstTable.from_dataframe(df)
    restored = _roundtrip("burst_table", payload).to_dataframe()

    assert restored["First File"].tolist() == ["a.ptu", "", "c.ptu"]


def test_burst_selection_roundtrip_with_ranges_and_criteria():
    payload = BurstSelection(
        source_artifact_id="burst_table_1",
        burst_ids=np.array([10, 11], dtype=np.int64),
        start_indices=np.array([100, 500], dtype=np.uint64),
        stop_indices=np.array([180, 620], dtype=np.uint64),
        criteria={"min_photons": 50, "channels": [0, 1]},
        labels=["selected", "selected"],
    )

    decoded = _roundtrip("burst_selection", payload)

    assert decoded.source_artifact_id == "burst_table_1"
    np.testing.assert_array_equal(decoded.start_indices, payload.start_indices)
    np.testing.assert_array_equal(decoded.stop_indices, payload.stop_indices)
    assert decoded.criteria["min_photons"] == 50


def test_burst_selection_requires_selection_representation():
    payload = BurstSelection(source_artifact_id="burst_table_1")

    with pytest.raises(PayloadSchemaError, match="requires"):
        encode_payload("burst_selection", payload)


def test_burst_selection_range_shapes_must_match():
    payload = BurstSelection(
        start_indices=np.array([1, 2], dtype=np.uint64),
        stop_indices=np.array([3], dtype=np.uint64),
    )

    with pytest.raises(PayloadSchemaError, match="same shape"):
        encode_payload("burst_selection", payload)


def test_tttr_reference_roundtrip_with_header_json_and_tags():
    payload = TttrReference(
        source_artifact_id="raw_1",
        vendor_format="ptu",
        n_records=10,
        routing_channels=[0, 1],
        macro_time_resolution_s=1e-8,
        micro_time_resolution_s=1e-12,
        header_json={"CreatorSW_Name": "tttrlib", "Tags": {"MeasDesc_Resolution": 1e-12}},
        tags={"TTResult_NumberOfRecords": 10},
    )

    decoded = _roundtrip("tttr_reference", payload)

    assert decoded.header_json["Tags"]["MeasDesc_Resolution"] == 1e-12
    assert decoded.tags["TTResult_NumberOfRecords"] == 10


def test_tttr_photon_stream_roundtrip_with_ptu_metadata():
    payload = TttrPhotonStream(
        macro_times=np.array([100, 120, 140], dtype=np.uint64),
        micro_times=np.array([12, 13, 14], dtype=np.uint32),
        routing_channels=np.array([0, 1, 0], dtype=np.uint16),
        event_types=np.array([0, 0, 1], dtype=np.uint8),
        macro_time_resolution_s=1e-8,
        micro_time_resolution_s=1e-12,
        source_artifact_id="raw_ptu_1",
        vendor_format="ptu",
        record_type="rtPicoHarpT3",
        container_type="PQ_PTU",
        header_json={"CreatorSW_Name": "tttrlib", "Tags": {"MeasDesc_Resolution": 1e-12}},
        tags={"TTResult_NumberOfRecords": 3},
    )

    decoded = _roundtrip("tttr_photon_stream", payload)

    np.testing.assert_array_equal(decoded.macro_times, payload.macro_times)
    np.testing.assert_array_equal(decoded.micro_times, payload.micro_times)
    np.testing.assert_array_equal(decoded.routing_channels, payload.routing_channels)
    np.testing.assert_array_equal(decoded.event_types, payload.event_types)
    assert decoded.header_json["Tags"]["MeasDesc_Resolution"] == 1e-12
    assert decoded.tags["TTResult_NumberOfRecords"] == 3


def test_tttr_photon_stream_event_arrays_must_match():
    payload = TttrPhotonStream(
        macro_times=np.array([100, 120], dtype=np.uint64),
        micro_times=np.array([12], dtype=np.uint32),
        routing_channels=np.array([0, 1], dtype=np.uint16),
        macro_time_resolution_s=1e-8,
        micro_time_resolution_s=1e-12,
    )

    with pytest.raises(PayloadSchemaError, match="micro_times"):
        encode_payload("tttr_photon_stream", payload)


def test_generic_curve_curve_like_roundtrip():
    class CurveLike:
        """Minimal object exposing the arrays archived by ``GenericCurve``."""

        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([3.0, 4.0], dtype=np.float64)
        ex = np.array([0.1, 0.2], dtype=np.float64)
        ey = np.array([0.3, 0.4], dtype=np.float64)
        mask = np.array([True, False], dtype=bool)

    curve = CurveLike()
    payload = GenericCurve.from_curve_like(curve)
    decoded = _roundtrip("generic_curve", payload)
    restored = decoded.to_curve_kwargs()

    np.testing.assert_array_equal(restored["x"], curve.x)
    np.testing.assert_array_equal(restored["y"], curve.y)
    np.testing.assert_array_equal(restored["mask"], curve.mask)


def test_missing_required_field_raises_schema_error():
    with pytest.raises(PayloadSchemaError, match="required|schema"):
        encode_payload("fcs_correlation", {"lag": np.array([1.0], dtype=np.float64)})


def test_wrong_dtype_raises_schema_error():
    payload = FcsCorrelation(
        lag=np.array([1.0], dtype=np.float32),
        correlation=np.array([1.0], dtype=np.float64),
    )

    with pytest.raises(PayloadSchemaError, match="dtype"):
        encode_payload("fcs_correlation", payload)


def test_wrong_shape_raises_schema_error():
    payload = Spectrum(
        wavelength=np.array([500.0, 510.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
        spectrum_type="emission",
    )

    with pytest.raises(PayloadSchemaError, match="shape"):
        encode_payload("spectra", payload)


def test_generic_table_rejects_object_dtype():
    payload = BurstTable(
        columns=["model"],
        dtypes=["object"],
        data={"model": np.array([{"a": 1}], dtype=object)},
    )

    with pytest.raises(PayloadSchemaError, match="object dtype"):
        encode_payload("burst_table", payload)


def test_fcs_correlation_lag_must_match_last_axis():
    payload = FcsCorrelation(
        lag=np.array([1e-6, 2e-6, 3e-6], dtype=np.float64),
        correlation=np.ones((2, 2), dtype=np.float64),
    )

    with pytest.raises(PayloadSchemaError, match="lag length"):
        encode_payload("fcs_correlation", payload)


def test_fcs_correlation_rejects_unsupported_dimensions():
    payload = FcsCorrelation(
        lag=np.array([1e-6, 2e-6], dtype=np.float64),
        correlation=np.ones((1, 1, 2), dtype=np.float64),
    )

    with pytest.raises(PayloadSchemaError, match="one- or two-dimensional"):
        encode_payload("fcs_correlation", payload)


def test_fcs_curve_names_must_match_curve_count():
    payload = FcsCorrelation(
        lag=np.array([1e-6, 2e-6], dtype=np.float64),
        correlation=np.ones((2, 2), dtype=np.float64),
        curve_names=["only_one"],
    )

    with pytest.raises(PayloadSchemaError, match="curve_names"):
        encode_payload("fcs_correlation", payload)


def test_unknown_kind_in_decode_raises_schema_error():
    blob = msgpack.packb(
        {"v": 1, "kind": "unknown_kind", "schema": 1, "units": {}, "meta": {}, "body": {}},
        use_bin_type=True,
    )

    with pytest.raises(PayloadSchemaError, match="unknown payload kind"):
        decode_payload(blob)


def test_schema_version_migration_path(monkeypatch):
    payload = GenericCurve(
        x=np.array([1.0], dtype=np.float64),
        y=np.array([2.0], dtype=np.float64),
    )
    blob, _ = encode_payload("generic_curve", payload)
    envelope = msgpack.unpackb(blob, raw=False)

    def upgrade(envelope):
        envelope["meta"]["upgraded"] = True
        return envelope

    monkeypatch.setattr(GenericCurve, "SCHEMA_VERSION", 2)
    MIGRATIONS[("generic_curve", 1, 2)] = upgrade
    try:
        decoded = decode_payload(msgpack.packb(envelope, use_bin_type=True))
    finally:
        MIGRATIONS.pop(("generic_curve", 1, 2), None)

    np.testing.assert_array_equal(decoded.x, payload.x)


def test_blob_size_sanity_for_large_fcs_curve():
    n_points = 10_000
    payload = FcsCorrelation(
        lag=np.linspace(1e-6, 1.0, n_points, dtype=np.float64),
        correlation=np.ones(n_points, dtype=np.float64),
    )

    blob, _ = encode_payload("fcs_correlation", payload)

    assert len(blob) < n_points * 8 * 3
