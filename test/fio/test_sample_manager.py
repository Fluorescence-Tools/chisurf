"""Tests for MFDB sample management."""
from __future__ import annotations

import os
import tempfile

import pytest

from chisurf.core.data import DataCurve
from chisurf.core.experiments.core.reader import ExperimentReader
from chisurf.core.mfdb.models import SampleDefinition
from chisurf.core.mfdb.project_archiver import archive_project_to_mfdb
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.sample_manager import (
    create_sample,
    find_sample_by_name,
    get_artifacts_for_sample,
    get_sample,
    get_sample_for_artifact,
    get_sample_full_description,
    link_artifact_to_sample,
    list_samples,
)


@pytest.fixture
def db():
    """Create a temporary MFDatabase for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        database = MFDatabase(os.path.join(tmpdir, "test.db"))
        try:
            yield database
        finally:
            database.close()


def test_create_and_get_sample(db):
    """Creating a sample stores metadata and returns it by ID."""
    definition = SampleDefinition(
        name="HP3-Cy3B-ATTO647N",
        description="DNA hairpin with Cy3B and ATTO647N",
        entity_name="DNA hairpin HP3",
        entity_type="dna",
        donor_probe_name="Cy3B",
        acceptor_probe_name="ATTO647N",
        buffer_description="PBS pH 7.4",
        ph=7.4,
        temperature_k=298.15,
        salt_concentration_m=0.15,
    )
    sample_id = create_sample(db, definition)

    sample = get_sample(db, sample_id)

    assert sample is not None
    assert sample["name"] == "HP3-Cy3B-ATTO647N"
    assert sample["entity_name"] == "DNA hairpin HP3"
    assert sample["donor_probe_name"] == "Cy3B"
    assert sample["acceptor_probe_name"] == "ATTO647N"
    assert sample["ph"] == 7.4
    assert sample["temperature_k"] == 298.15
    assert sample["salt_concentration_m"] == 0.15


def test_create_sample_idempotent(db):
    """Creating the same sample name twice returns the existing sample ID."""
    definition = SampleDefinition(name="test_sample")

    first = create_sample(db, definition)
    second = create_sample(db, definition)

    assert first == second


def test_create_sample_delegates_to_orm_adapter(db, monkeypatch):
    """Public sample creation goes through the SQLAlchemy graph adapter."""
    calls = []

    def fake_create_sample_graph(db_arg, definition_arg, **kwargs):
        calls.append((db_arg, definition_arg, kwargs))
        return kwargs["sample_id"]

    monkeypatch.setattr(
        "chisurf.core.mfdb.orm.sample_repository.create_sample_graph",
        fake_create_sample_graph,
    )

    definition = SampleDefinition(name="Adapter Sample")
    sample_id = create_sample(db, definition)

    assert sample_id == "adapter_sample"
    assert len(calls) == 1
    db_arg, definition_arg, kwargs = calls[0]
    assert db_arg is db
    assert definition_arg is definition
    assert kwargs["sample_id"] == "adapter_sample"
    assert kwargs["display_name"] == "Adapter Sample"
    assert kwargs["metadata_json"]


def test_get_sample_full_description_uses_orm_graph(db, monkeypatch):
    """Public full-description reads use the SQLAlchemy graph adapter."""
    calls = []

    def fake_get_sample_graph(db_arg, sample_id):
        calls.append((db_arg, sample_id))
        return {
            "sample": {
                "sample_id": sample_id,
                "description": "from orm graph",
                "solvent_phase": "liquid",
            },
            "entities": [],
            "probes": [],
            "condition": None,
            "fret_pairs": [],
            "key_values": [],
        }

    monkeypatch.setattr(
        "chisurf.core.mfdb.orm.sample_repository.get_sample_graph",
        fake_get_sample_graph,
    )

    description = get_sample_full_description(db, "sample_1")

    assert description is not None
    assert description["description"] == "from orm graph"
    assert description["sample_id"] == "sample_1"
    assert calls == [(db, "sample_1")]


def test_create_sample_with_optional_none(db):
    """Creating a sample with Optional fields set to None works correctly."""
    definition = SampleDefinition(
        name="minimal_sample",
        ph=None,
        temperature_k=None,
        salt_concentration_m=None,
        donor_position=None,
        acceptor_position=None,
    )
    sample_id = create_sample(db, definition)

    sample = get_sample(db, sample_id)

    assert sample is not None
    assert sample["ph"] is None
    assert sample["temperature_k"] is None
    assert sample["salt_concentration_m"] is None
    assert sample["donor_position"] is None
    assert sample["acceptor_position"] is None


def test_list_samples(db):
    """list_samples returns active samples ordered by display name."""
    create_sample(db, SampleDefinition(name="sample_b"))
    create_sample(db, SampleDefinition(name="sample_a"))

    samples = list_samples(db)

    assert [sample["name"] for sample in samples] == ["sample_a", "sample_b"]


def test_find_sample_by_name(db):
    """find_sample_by_name returns the matching sample ID or None."""
    sample_id = create_sample(db, SampleDefinition(name="find_me"))

    assert find_sample_by_name(db, "find_me") == sample_id
    assert find_sample_by_name(db, "nope") is None


def test_link_artifact_to_sample(db):
    """Artifacts can be linked to samples through mfdb_edge."""
    sample_id = create_sample(db, SampleDefinition(name="linked_sample"))
    artifact_id = "art_001"
    db.register_artifact(
        artifact_id=artifact_id,
        artifact_kind="raw_measurement",
        data_format="ptu",
        storage_mode="local_file",
        file_path="/tmp/measurement.ptu",
    )

    link_artifact_to_sample(db, artifact_id, sample_id)
    link_artifact_to_sample(db, artifact_id, sample_id)

    assert get_sample_for_artifact(db, artifact_id) == sample_id
    assert get_artifacts_for_sample(db, sample_id) == [artifact_id]


def test_archive_project_links_dataset_artifacts_to_sample(db, tmp_path):
    """Project archiving links source and processed-data artifacts to the dataset sample."""
    sample_id = create_sample(db, SampleDefinition(name="archived_sample"))
    source_path = tmp_path / "measurement.ptu"
    source_path.write_text("dummy", encoding="utf-8")
    payload = {
        "meta": {},
        "datasets": {
            "ds_1": {
                "name": "curve",
                "filename": str(source_path),
                "x": [0.0, 1.0],
                "y": [1.0, 2.0],
                "ex": [0.0, 0.0],
                "ey": [1.0, 1.0],
                "meta_data": {"sample_id": sample_id},
            }
        },
    }

    summary = archive_project_to_mfdb(
        db,
        payload,
        version_id="ver_1",
        project_id="proj_1",
        version_number=1,
    )

    dataset_artifact_id = summary["dataset_artifacts"][0]
    linked_artifacts = get_artifacts_for_sample(db, sample_id)
    assert dataset_artifact_id in linked_artifacts
    assert any(artifact_id.startswith("src_") for artifact_id in linked_artifacts)


class _DummyReader(ExperimentReader):
    """Tiny reader used to verify sample metadata stamping."""

    def __init__(self, *args, **kwargs):
        """Initialize the dummy reader."""
        super().__init__(*args, **kwargs)
        self.experiment = None

    def autofitrange(self, data, **kwargs):
        """Return the full curve range."""
        return 0, len(data)

    def read(self, filename=None, **kwargs):
        """Return a simple data curve."""
        return DataCurve(x=[0.0, 1.0], y=[1.0, 2.0])


def test_reader_stamps_sample_id_into_dataset_metadata():
    """ExperimentReader stamps selected sample_id into dataset metadata."""
    reader = _DummyReader(sample_id="sample_1", record_provenance=False)

    group = reader.get_data()
    curve = group[0]

    assert curve.meta_data["sample_id"] == "sample_1"


# ── Tests for SampleDefinition vocabulary validation ────────────────────────


def test_sample_definition_valid_vocabulary(db):
    """SampleDefinition with valid vocabulary values is accepted."""
    definition = SampleDefinition(
        name="valid_sample",
        entity_type="protein",
        donor_probe_name="Cy3B",
        acceptor_probe_name="ATTO647N",
        validate_vocabulary=True,
    )
    # Should not raise
    sample_id = create_sample(db, definition)
    assert sample_id


def test_sample_definition_invalid_entity_type_raises():
    """SampleDefinition with invalid entity_type raises ValueError."""
    from chisurf.core.mfdb.models import ENTITY_TYPES

    invalid_type = "invalid_entity_type_12345"
    assert invalid_type not in ENTITY_TYPES

    with pytest.raises(ValueError, match="Invalid entity_type"):
        SampleDefinition(
            name="invalid_sample",
            entity_type=invalid_type,
            validate_vocabulary=True,
        )


def test_sample_definition_invalid_probe_name_warns(caplog):
    """SampleDefinition with invalid probe name logs warning but does not raise.

    Per PRD-02: unknown probe names should WARN (not reject), since custom
    dyes are valid. Only entity_type should hard-reject.
    """
    from chisurf.core.mfdb.models import COMMON_PROBE_NAMES
    import logging

    invalid_probe = "invalid_probe_xyz"
    assert invalid_probe not in COMMON_PROBE_NAMES

    # Should not raise - only logs a warning
    with caplog.at_level(logging.WARNING):
        definition = SampleDefinition(
            name="invalid_sample",
            donor_probe_name=invalid_probe,
            validate_vocabulary=True,
        )
        assert "not in COMMON_PROBE_NAMES" in caplog.text


def test_sample_definition_validation_disabled():
    """SampleDefinition with invalid values is accepted when validation is disabled."""
    definition = SampleDefinition(
        name="no_validation",
        entity_type="invalid_type",
        donor_probe_name="invalid_probe",
        validate_vocabulary=False,  # Default is False for backward compatibility
    )
    # Should not raise
    assert definition.name == "no_validation"


# ── Tests for Sample Requests ──────────────────────────────────────────────


def test_sample_create_request_to_definition():
    """SampleCreateRequest can be converted to SampleDefinition."""
    from chisurf.core.mfdb.sample_requests import SampleCreateRequest

    request = SampleCreateRequest(
        name="request_sample",
        description="A sample from request",
        entity_type="dna",
        donor_probe_name="Cy3B",
        ph=7.4,
    )

    definition = request.to_sample_definition()

    assert definition.name == "request_sample"
    assert definition.description == "A sample from request"
    assert definition.entity_type == "dna"
    assert definition.donor_probe_name == "Cy3B"
    assert definition.ph == 7.4


def test_sample_create_request_validates_vocabulary():
    """SampleCreateRequest validates vocabulary by default."""
    from chisurf.core.mfdb.sample_requests import SampleCreateRequest

    with pytest.raises(ValueError, match="Invalid entity_type"):
        SampleCreateRequest(
            name="invalid",
            entity_type="invalid_type",
            validate_vocabulary=True,
        )


def test_sample_create_request_requires_name():
    """SampleCreateRequest requires a name."""
    from chisurf.core.mfdb.sample_requests import SampleCreateRequest

    with pytest.raises(ValueError, match="sample name is required"):
        SampleCreateRequest(name="")

    with pytest.raises(ValueError, match="sample name is required"):
        SampleCreateRequest(name="   ")


def test_sample_update_request_requires_sample_id():
    """SampleUpdateRequest requires a sample_id."""
    from chisurf.core.mfdb.sample_requests import SampleUpdateRequest

    with pytest.raises(ValueError, match="sample_id is required"):
        SampleUpdateRequest(sample_id="")


def test_sample_link_request_validates_fields():
    """SampleLinkRequest validates required fields."""
    from chisurf.core.mfdb.sample_requests import SampleLinkRequest

    with pytest.raises(ValueError, match="artifact_id is required"):
        SampleLinkRequest(artifact_id="", sample_id="sample_1")

    with pytest.raises(ValueError, match="sample_id is required"):
        SampleLinkRequest(artifact_id="art_1", sample_id="")


def test_sample_query_request_validates_limits():
    """SampleQueryRequest validates limit and offset."""
    from chisurf.core.mfdb.sample_requests import SampleQueryRequest

    with pytest.raises(ValueError, match="limit must be at least 1"):
        SampleQueryRequest(limit=0)

    with pytest.raises(ValueError, match="limit must be at least 1"):
        SampleQueryRequest(limit=-1)

    with pytest.raises(ValueError, match="offset must be non-negative"):
        SampleQueryRequest(offset=-1)


def test_create_sample_with_fret_pairs_and_positions(db):
    """Regression test for R16-1: structured sample with FretPairDefinition.

    Creates a sample with multiple probes and FRET pairs, verifies that:
    1. Sample creation succeeds without SQL errors
    2. All FRET pair data is persisted
    3. Full description includes forster_radius_id, sample_id, and scoped FRET pairs
    """
    from chisurf.core.mfdb.models import (
        EntityDefinition,
        FretPairDefinition,
        ProbeDefinition,
    )
    from chisurf.core.mfdb.sample_manager import get_sample_full_description

    # Create a 3-color FRET sample with explicit entities, probes, and FRET pairs
    definition = SampleDefinition(
        name="3color_fret_sample",
        description="3-color FRET sample for testing",
        entities=[
            EntityDefinition(
                name="T4 Lysozyme",
                entity_type="protein",
                sequence=list("MNGTELK")  # truncated for test
            ),
        ],
        probes=[
            ProbeDefinition(name="Cy3B", seq_id=1, asym_id="A", entity_index=0),
            ProbeDefinition(name="ATTO550", seq_id=10, asym_id="A", entity_index=0),
            ProbeDefinition(name="ATTO647N", seq_id=20, asym_id="A", entity_index=0),
        ],
        fret_pairs=[
            FretPairDefinition(
                probe_1_index=0,
                probe_2_index=1,
                forster_radius_nm=5.4,
                kappa_squared=2.0/3.0,
                refractive_index=1.4,
            ),
            FretPairDefinition(
                probe_1_index=1,
                probe_2_index=2,
                forster_radius_nm=6.2,
                kappa_squared=2.0/3.0,
                refractive_index=1.4,
            ),
        ],
        buffer_description="PBS pH 7.4",
        ph=7.4,
        temperature_k=298.15,
    )

    # This should not raise sqlite3.OperationalError: 12 values for 13 columns
    sample_id = create_sample(db, definition)

    # Verify sample was created
    assert sample_id is not None
    assert sample_id == "3color_fret_sample"

    # Get full description and verify FRET pairs are persisted
    full_desc = get_sample_full_description(db, sample_id)
    assert full_desc is not None

    # Verify FRET pairs exist
    assert "fret_pairs" in full_desc
    fret_pairs = full_desc["fret_pairs"]
    assert len(fret_pairs) == 2

    # Verify each FRET pair has the expected data
    for pair in fret_pairs:
        assert "forster_radius_id" in pair
        assert pair["forster_radius_id"] is not None
        assert "forster_radius_nm" in pair
        assert pair["forster_radius_nm"] in [5.4, 6.2]
        assert "donor_probe" in pair
        assert "acceptor_probe" in pair
        assert "sample_id" in pair or pair.get("id") is not None

    # Verify probes are present
    assert "probes" in full_desc
    assert len(full_desc["probes"]) == 3

    # Verify entities are present
    assert "entities" in full_desc
    assert len(full_desc["entities"]) == 1
