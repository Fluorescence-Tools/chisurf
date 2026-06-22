"""Tests for Burst Selection MFDB registration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from chisurf.core.mfdb.models import SampleDefinition
from chisurf.core.mfdb.payload_models import BurstTable
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.result_registry import read_result, register_raw_measurement, set_global_db
from chisurf.core.mfdb.sample_manager import create_sample, get_artifacts_for_sample
from chisurf.plugins.burst.burst_selection.api import selection as selection_module
from chisurf.plugins.burst.burst_selection.api.mfdb import BurstMFDBPipeline
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
    setup_id_for_name,
)
from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisSettings,
)


@pytest.fixture
def db():
    """Create a temporary MFDB for burst-pipeline tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        database = MFDatabase(os.path.join(tmpdir, "test.db"))
        try:
            yield database
        finally:
            set_global_db(None)
            database.close()


def _write_input(tmp_path: Path, name: str = "m000.spc") -> Path:
    """Create a fake TTTR input file.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.
    name : str
        File name.

    Returns
    -------
    Path
        Created input path.

    """
    path = tmp_path / name
    path.write_bytes(b"fake photons")
    return path


def _write_bur(tmp_path: Path, name: str = "m000.bur") -> Path:
    """Create a minimal burst table file.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.
    name : str
        File name.

    Returns
    -------
    Path
        Created ``.bur`` path.

    """
    path = tmp_path / name
    path.write_text("First Photon\tLast Photon\tNumber of Photons\n0\t10\t11\n")
    return path


def _request(input_path: Path, *, sample_id: str = "", setup_id: str = "") -> AnalysisRequest:
    """Build a registration request for one input file.

    Parameters
    ----------
    input_path : Path
        Input TTTR file path.
    sample_id : str
        Optional sample ID.
    setup_id : str
        Optional MFDB setup ID.

    Returns
    -------
    AnalysisRequest
        Request with deterministic settings.

    """
    request = AnalysisRequest(files=[str(input_path)], settings=AnalysisSettings(output_formats=[]))
    request.mfdb.sample_id = sample_id
    request.mfdb.setup_id = setup_id
    return request


def _result(input_path: Path, *, bur_path: Path | None = None) -> AnalysisResult:
    """Build a synthetic analysis result for one input file.

    Parameters
    ----------
    input_path : Path
        Input TTTR file path.
    bur_path : Path, optional
        Per-file burst output path.

    Returns
    -------
    AnalysisResult
        Analysis result with one burst row.

    """
    normalized = str(input_path.resolve())
    output_paths_by_file = {normalized: {"bur": str(bur_path)}} if bur_path else {normalized: {}}
    output_paths = {"bur": str(bur_path)} if bur_path else {}
    return AnalysisResult(
        files=[str(input_path)],
        dataframes={str(input_path): [{"First Photon": 0, "Last Photon": 10, "Number of Photons": 11}]},
        output_paths=output_paths,
        output_paths_by_file=output_paths_by_file,
        metadata={"n_files": 1, "n_bursts": 1, "n_selected": 11, "n_photons": 20},
    )


def test_registers_raw_input_when_source_missing(db, tmp_path: Path) -> None:
    """Pipeline registers one raw input artifact when no source ID is supplied."""
    input_path = _write_input(tmp_path)
    bur_path = _write_bur(tmp_path)

    registration = BurstMFDBPipeline(db).register_run(_request(input_path), _result(input_path, bur_path=bur_path))

    assert registration.input_artifacts[str(input_path.resolve())]
    rows = db.list_artifacts(artifact_kind="raw_measurement")
    assert len(rows) == 1
    assert rows[0]["data_format"] == "spc"


def test_reuses_source_artifact_without_duplicate_raw_input(db, tmp_path: Path) -> None:
    """Caller-provided source artifacts are reused instead of duplicated."""
    input_path = _write_input(tmp_path)
    bur_path = _write_bur(tmp_path)
    source_id = register_raw_measurement(str(input_path), db=db)
    request = _request(input_path)
    request.mfdb.source_artifact_ids = {str(input_path.resolve()): source_id}

    registration = BurstMFDBPipeline(db).register_run(request, _result(input_path, bur_path=bur_path))

    assert registration.input_artifacts[str(input_path.resolve())] == source_id
    assert len(db.list_artifacts(artifact_kind="raw_measurement")) == 1


def test_registers_burst_table_artifact_for_bur_output(db, tmp_path: Path) -> None:
    """Per-file ``.bur`` outputs register as burst_table artifacts."""
    input_path = _write_input(tmp_path)
    bur_path = _write_bur(tmp_path)

    registration = BurstMFDBPipeline(db).register_run(_request(input_path), _result(input_path, bur_path=bur_path))
    artifact = db.get_artifact(registration.burst_table_artifacts[str(input_path.resolve())])

    assert artifact["artifact_kind"] == "burst_table"
    assert artifact["data_format"] == "bur"


def test_registers_typed_burst_table_from_rows_when_no_bur_path(db, tmp_path: Path) -> None:
    """Burst rows are stored as a typed msgpack BurstTable when no file exists."""
    input_path = _write_input(tmp_path)

    registration = BurstMFDBPipeline(db).register_run(_request(input_path), _result(input_path))
    payload = read_result(db, registration.burst_table_artifacts[str(input_path.resolve())])

    assert isinstance(payload, BurstTable)
    assert payload.columns == ["First Photon", "Last Photon", "Number of Photons"]


def test_burst_table_derives_from_matching_raw_input(db, tmp_path: Path) -> None:
    """Burst tables receive a derived_from edge to their matching raw input."""
    input_path = _write_input(tmp_path)

    registration = BurstMFDBPipeline(db).register_run(_request(input_path), _result(input_path))
    conn = getattr(db, "conn")
    row = conn.execute(
        """SELECT relationship_type FROM mfdb_edge
           WHERE source_node_id = ? AND target_node_id = ? AND deleted_at IS NULL""",
        (
            registration.burst_table_artifacts[str(input_path.resolve())],
            registration.input_artifacts[str(input_path.resolve())],
        ),
    ).fetchone()

    assert row["relationship_type"] == "derived_from"


def test_registered_artifacts_link_to_sample_id(db, tmp_path: Path) -> None:
    """Raw and burst-table artifacts are linked to the supplied sample."""
    input_path = _write_input(tmp_path)
    sample_id = create_sample(db, SampleDefinition(name="sample A"))

    registration = BurstMFDBPipeline(db).register_run(
        _request(input_path, sample_id=sample_id),
        _result(input_path),
    )
    linked = set(get_artifacts_for_sample(db, sample_id))

    assert registration.input_artifacts[str(input_path.resolve())] in linked
    assert registration.burst_table_artifacts[str(input_path.resolve())] in linked


def test_records_scalar_burst_parameters(db, tmp_path: Path) -> None:
    """Burst selection parameters are recorded in mfdb_parameter."""
    input_path = _write_input(tmp_path)

    registration = BurstMFDBPipeline(db).register_run(_request(input_path), _result(input_path))
    conn = getattr(db, "conn")
    rows = conn.execute(
        """SELECT p.name, p.value FROM mfdb_parameter p
           JOIN mfdb_operation_artifact oa ON oa.operation_id = p.operation_id
           WHERE oa.artifact_id = ? AND oa.direction = 'output'""",
        (registration.burst_table_artifacts[str(input_path.resolve())],),
    ).fetchall()
    values = {row["name"]: row["value"] for row in rows}

    assert values["min_photons"] == 60.0
    assert values["photon_window"] == 10.0


def test_burst_table_operation_links_selected_setup(db, tmp_path: Path) -> None:
    """Burst table operations should reference the selected detector setup."""
    input_path = _write_input(tmp_path)
    setup_id = "tttr_detector_setup:bh_spc_130"
    db.save_setup(
        setup_id=setup_id,
        name="BH SPC-130",
        configuration={"setup_type": "tttr_detector_setup"},
        detectors={"green": {"chs": [0, 8]}, "red": {"chs": [1, 9]}},
    )

    registration = BurstMFDBPipeline(db).register_run(_request(input_path, setup_id=setup_id), _result(input_path))
    conn = getattr(db, "conn")
    row = conn.execute(
        """SELECT op.setup_id FROM mfdb_operation op
           JOIN mfdb_operation_artifact oa ON oa.operation_id = op.operation_id
           WHERE oa.artifact_id = ? AND oa.direction = 'output'""",
        (registration.burst_table_artifacts[str(input_path.resolve())],),
    ).fetchone()

    assert row["setup_id"] == setup_id


def test_unavailable_mfdb_reports_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Missing MFDB does not crash registration and returns warnings."""
    input_path = _write_input(tmp_path)
    monkeypatch.setattr("chisurf.core.mfdb.result_registry._get_global_db", lambda: None)
    set_global_db(None)

    registration = BurstMFDBPipeline().register_run(_request(input_path), _result(input_path))

    assert registration.input_artifacts == {}
    assert registration.burst_table_artifacts == {}
    assert registration.warnings


def test_two_input_files_keep_per_file_output_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Analysis merging preserves per-file output paths for repeated roles."""
    first = _write_input(tmp_path, "first.spc")
    second = _write_input(tmp_path, "second.spc")

    def fake_analyze_file(path: str, **kwargs: object) -> AnalysisResult:
        normalized = str(Path(path).resolve())
        bur_path = tmp_path / f"{Path(path).stem}.bur"
        bur_path.write_text("First Photon\tLast Photon\n0\t1\n")
        return AnalysisResult(
            files=[str(path)],
            dataframes={str(path): [{"First Photon": 0, "Last Photon": 1}]},
            output_paths={"bur": str(bur_path)},
            output_paths_by_file={normalized: {"bur": str(bur_path)}},
            metadata={"n_photons": 1, "n_selected": 1, "n_bursts": 1},
        )

    monkeypatch.setattr(selection_module, "analyze_file", fake_analyze_file)

    result = selection_module.analyze_request(
        AnalysisRequest(
            files=[str(first), str(second)],
            settings=AnalysisSettings(output_formats=["bur"]),
            output_dir=str(tmp_path),
        )
    )

    assert result.output_paths_by_file[str(first.resolve())]["bur"].endswith("first.bur")
    assert result.output_paths_by_file[str(second.resolve())]["bur"].endswith("second.bur")


def test_selected_setup_derives_setup_id_when_empty(db, tmp_path: Path, monkeypatch) -> None:
    """A burst run with selected_setup (but no setup_id) derives the setup id
    and populates mfdb_operation.setup_id."""
    from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
        _resolve_active_user_id,
    )
    user_id = "user_default"
    monkeypatch.setattr(
        "chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups._resolve_active_user_id",
        lambda: user_id,
    )
    input_path = _write_input(tmp_path)
    setup_id = setup_id_for_name("BH SPC-130", user_id=user_id)
    db.save_setup(
        setup_id=setup_id,
        name="BH SPC-130",
        configuration={"setup_type": "tttr_detector_setup"},
        detectors={"green": {"chs": [0, 8]}, "red": {"chs": [1, 9]}},
        created_by_user_id=user_id,
    )

    request = _request(input_path)
    request.mfdb.setup_id = ""
    request.selected_setup = "BH SPC-130"

    registration = BurstMFDBPipeline(db).register_run(request, _result(input_path))
    conn = getattr(db, "conn")
    row = conn.execute(
        """SELECT op.setup_id FROM mfdb_operation op
           JOIN mfdb_operation_artifact oa ON oa.operation_id = op.operation_id
           WHERE oa.artifact_id = ? AND oa.direction = 'output'""",
        (registration.burst_table_artifacts[str(input_path.resolve())],),
    ).fetchone()

    assert row["setup_id"] == setup_id


def test_nonexistent_setup_emits_warning_and_leaves_setup_id_null(
    db, tmp_path: Path
) -> None:
    """A burst run naming a non-existent setup still succeeds, emits a warning,
    and leaves mfdb_operation.setup_id NULL."""
    input_path = _write_input(tmp_path)

    request = _request(input_path)
    request.mfdb.setup_id = "tttr_detector_setup:does_not_exist"

    registration = BurstMFDBPipeline(db).register_run(request, _result(input_path))

    assert any("does not exist" in w for w in registration.warnings)

    conn = getattr(db, "conn")
    row = conn.execute(
        """SELECT op.setup_id FROM mfdb_operation op
           JOIN mfdb_operation_artifact oa ON oa.operation_id = op.operation_id
           WHERE oa.artifact_id = ? AND oa.direction = 'output'""",
        (registration.burst_table_artifacts[str(input_path.resolve())],),
    ).fetchone()
    assert row["setup_id"] is None


def test_invalid_sample_registration_leaves_no_artifact_rows(db, tmp_path: Path) -> None:
    """Invalid sample IDs are reported without partial artifact/object rows."""
    input_path = _write_input(tmp_path)

    registration = BurstMFDBPipeline(db).register_run(
        _request(input_path, sample_id="missing_sample"),
        _result(input_path),
    )

    assert registration.input_artifacts == {}
    assert registration.burst_table_artifacts == {}
    assert registration.warnings
    conn = getattr(db, "conn")
    assert conn.execute("SELECT COUNT(*) FROM mfdb_artifact").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM mfdb_object").fetchone()[0] == 0
