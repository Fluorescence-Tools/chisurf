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
