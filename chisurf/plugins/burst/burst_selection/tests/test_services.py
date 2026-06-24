"""Tests for Burst Selection service registration."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.result_registry import set_global_db
from chisurf.plugins.burst.burst_selection.api.contract import (
    METHOD_ANALYZE_FILES,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FIT_GMM,
    METHOD_INSPECT_BUR,
    METHOD_LOAD_DIAGNOSTICS,
)
from chisurf.plugins.burst.burst_selection.backend import services as backend_services
from chisurf.plugins.burst.burst_selection.api.models import AnalysisResult
from chisurf.plugins.burst.burst_selection.server.services import (
    list_methods,
    register_burst_selection_services,
)
from chisurf.plugins.burst.burst_selection.backend.services import (
    contract_handler,
    list_methods as list_backend_methods,
)


def test_register_burst_selection_services() -> None:
    """Burst Selection services should register with ServiceDispatcher."""

    class Dispatcher:
        """Minimal dispatcher double."""

        def __init__(self) -> None:
            self.names = []

        def register(self, name: str, handler):
            self.names.append(name)

    dispatcher = Dispatcher()
    register_burst_selection_services(dispatcher)
    assert dispatcher.names == [
        METHOD_ANALYZE_FILES,
        METHOD_INSPECT_BUR,
        METHOD_FIT_GMM,
        METHOD_LOAD_DIAGNOSTICS,
        METHOD_DESCRIBE_CONTRACT,
    ]


def test_list_methods() -> None:
    """The method catalogue should expose all Burst Selection RPC methods."""
    methods = list_methods()
    assert METHOD_ANALYZE_FILES in methods
    assert METHOD_INSPECT_BUR in methods
    assert METHOD_FIT_GMM in methods
    assert METHOD_DESCRIBE_CONTRACT in methods
    # All methods have descriptions
    for name, desc in methods.items():
        assert isinstance(desc, str) and len(desc) > 0


def test_backend_contract_handler_returns_service_result() -> None:
    """Workflow clients should be able to discover the contract through RPC."""
    result = contract_handler()
    assert result["ok"] is True
    assert result["result"]["plugin_id"] == "burst_selection"
    assert "burst_selection.jobs.analyze_files" in result["result"]["rpc_methods"]


def test_backend_method_catalogue_includes_contract_method() -> None:
    """The backend catalogue should publish the contract method."""
    methods = list_backend_methods()
    assert methods["burst_selection.contract.describe"] == "Return the Burst Selection workflow contract."


def test_analyze_files_handler_returns_mfdb_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Service handler should append MFDB artifact IDs when registration succeeds."""
    input_path = tmp_path / "m000.spc"
    input_path.write_bytes(b"fake photons")

    def fake_analyze_request(request):
        normalized = str(input_path.resolve())
        return AnalysisResult(
            files=[str(input_path)],
            dataframes={str(input_path): [{"First Photon": 0, "Last Photon": 1}]},
            output_paths_by_file={normalized: {}},
            metadata={"n_files": 1, "n_bursts": 1},
        )

    db = MFDatabase(tmp_path / "test.db")
    set_global_db(db)
    monkeypatch.setattr(backend_services, "analyze_request", fake_analyze_request)
    try:
        response = backend_services.analyze_files_handler(
            files=[str(input_path)],
            settings={"output_formats": []},
        )
    finally:
        set_global_db(None)
        db.close()

    assert response["ok"] is True
    artifacts = response["result"]["mfdb_artifacts"]
    assert artifacts["input_artifacts"][str(input_path.resolve())]
    assert artifacts["burst_table_artifacts"][str(input_path.resolve())]


def test_multi_file_run_registers_single_grouped_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A multi-file run must appear as ONE named directory group, not one
    UUID-named directory per file.

    Every file's per-file directory sidecar points at the same shared run output
    folder, so registering them per file produced many duplicate
    ``external_reference``/``directory`` entries. Only the single run output
    folder should be registered, and it must surface a real name (the folder
    basename) rather than its artifact UUID.
    """
    from chisurf.gui.widgets.mfdb.dataset_browser import _get_display_name

    files = [tmp_path / f"m00{i}.spc" for i in range(3)]
    for f in files:
        f.write_bytes(b"fake photons")

    run_folder = tmp_path / "burstwise_All 0.1500#40"
    run_folder.mkdir()
    info_dir = run_folder / "Info"
    info_dir.mkdir()
    bur_dir = run_folder / "bi4_bur"
    bur_dir.mkdir()

    def fake_analyze_request(request):
        normalized = {str(f.resolve()): f for f in files}
        per_file = {}
        for norm, f in normalized.items():
            bur = bur_dir / f"{f.stem}.bur"
            bur.write_text("0 1\n")
            # Each file's per-file sidecars include the shared Info directory.
            per_file[norm] = {"bur": str(bur), "mti_dir": str(info_dir)}
        return AnalysisResult(
            files=[str(f) for f in files],
            dataframes={str(f): [{"First Photon": 0, "Last Photon": 1}] for f in files},
            output_paths_by_file=per_file,
            # The run output folder is the single group directory.
            output_paths={"output_folder": str(run_folder)},
            metadata={"n_files": len(files), "n_bursts": len(files)},
        )

    db = MFDatabase(tmp_path / "test.db")
    set_global_db(db)
    monkeypatch.setattr(backend_services, "analyze_request", fake_analyze_request)
    try:
        response = backend_services.analyze_files_handler(
            files=[str(f) for f in files],
            settings={"output_formats": []},
        )
        assert response["ok"] is True
        rows = db.conn.execute(
            "SELECT * FROM mfdb_artifact "
            "WHERE artifact_kind = 'external_reference' AND data_format = 'directory' "
            "AND deleted_at IS NULL"
        ).fetchall()
        dirs = [dict(r) for r in rows]
    finally:
        set_global_db(None)
        db.close()

    assert len(dirs) == 1, f"expected one grouped directory, got {len(dirs)}"
    assert _get_display_name(dirs[0]) == run_folder.name
