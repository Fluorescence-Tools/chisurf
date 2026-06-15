from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from click.testing import CliRunner
import numpy as np
import pytest

from chisurf.plugins.modelling.proteinmc.cli import cli
from chisurf.plugins.modelling.proteinmc.model import (
    DirectLabelingPotential,
    ProteinMCRunner,
    load_structure,
    normalize_settings,
)


PDB_148L = Path("test/data/atomic_coordinates/pdb_files/148l.pdb")


def _require_pdb2pqr() -> None:
    if shutil.which("pdb2pqr") is None and shutil.which("pdb2pqr30") is None:
        pytest.skip("ProteinMC structure preparation requires pdb2pqr")


def _labeling_json(structure, path: Path) -> Path:
    atoms = structure.atoms
    ca = np.where(np.array([_as_text(v) == "CA" for v in atoms["atom_name"]]))[0]
    assert ca.size >= 2
    a0 = atoms[ca[0]]
    a1 = atoms[ca[1]]
    distance = float(np.linalg.norm(a0["xyz"] - a1["xyz"]))
    payload = {
        "Positions": {
            "p1": _position(a0),
            "p2": _position(a1),
        },
        "Distances": {
            "p1_p2": {
                "Forster_radius": 52.0,
                "distance_type": "RDAMean",
                "position1_name": "p1",
                "position2_name": "p2",
                "distance": distance + 1.0,
                "error_neg": 1.0,
                "error_pos": 1.0,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _position(atom) -> dict:
    return {
        "atom_name": _as_text(atom["atom_name"]),
        "chain_identifier": _as_text(atom["chain"]),
        "residue_seq_number": int(atom["res_id"]),
        "residue_name": _as_text(atom["res_name"]),
        "attachment_atom_index": int(atom["i"]),
    }


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def test_normalize_settings_maps_legacy_keys() -> None:
    settings = normalize_settings({"n_out": 3, "pdbOut": 4, "movemap": [1, 0]})
    assert settings["pdb_nOut"] == 3
    assert settings["n_written"] == 4
    assert settings["move_map"] == [1, 0]


def test_load_structure_requires_pdb2pqr(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="requires pdb2pqr"):
        load_structure(PDB_148L)


def test_runner_settings_override_settings_file(tmp_path: Path) -> None:
    _require_pdb2pqr()
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"n_iter": 10, "pdbOut": 10}), encoding="utf-8")
    runner = ProteinMCRunner(
        PDB_148L,
        settings={"n_iter": 2, "pdbOut": 1, "potentials": []},
        settings_file=settings_file,
        output_file=tmp_path / "out.rmf3",
    )
    assert runner.settings["n_iter"] == 2
    assert runner.settings["n_written"] == 1


def test_direct_labeling_potential_scores_148l(tmp_path: Path) -> None:
    _require_pdb2pqr()
    structure = load_structure(PDB_148L)
    labeling_file = _labeling_json(structure, tmp_path / "labeling.fps.json")
    potential = DirectLabelingPotential(structure, labeling_file)
    assert potential.getEnergy() == pytest.approx(1.0, rel=1e-6)


def test_runner_writes_rmf3_readable_by_chimol(tmp_path: Path) -> None:
    _require_pdb2pqr()
    pytest.importorskip("RMF")
    pytest.importorskip("IMP")
    from chisurf.plugins.chimol.chimol.io.rmf import load_rmf_full

    structure = load_structure(PDB_148L)
    labeling_file = _labeling_json(structure, tmp_path / "labeling.fps.json")
    output = tmp_path / "proteinmc.rmf3"
    runner = ProteinMCRunner(
        PDB_148L,
        labeling_file=labeling_file,
        settings={"n_iter": 3, "n_out": 1, "pdbOut": 1, "potentials": []},
        output_file=output,
    )
    result = runner.run()
    assert Path(result.output_file).exists()
    data = load_rmf_full(output)
    assert data["frames"].shape[0] >= 1
    assert data["frames"].shape[2] == 3
    assert "Total_Score" in data["rmf_frame_series"]
    assert data["rmf_frame_series"]["Total_Score"].size == data["frames"].shape[0]


def test_rmf_writer_replaces_blank_chain_ids(tmp_path: Path) -> None:
    _require_pdb2pqr()
    pytest.importorskip("RMF")
    pytest.importorskip("IMP")
    from chisurf.plugins.modelling.proteinmc.rmf import ProteinMCRmfWriter

    structure = load_structure(PDB_148L)
    structure.atoms["chain"] = b""
    output = tmp_path / "blank_chain.rmf3"
    writer = ProteinMCRmfWriter(output, structure)
    writer.append(structure.xyz)
    writer.close()
    assert output.exists()


def test_rmf_writer_writes_pmi_stat_metadata(tmp_path: Path) -> None:
    pytest.importorskip("RMF")
    pytest.importorskip("IMP")
    from chisurf.plugins.chimol.chimol.io.rmf import load_rmf_full
    from chisurf.plugins.modelling.proteinmc.rmf import ProteinMCRmfWriter

    atoms = np.zeros(
        2,
        dtype={
            "names": ["xyz", "chain", "res_id", "res_name", "atom_name", "radius"],
            "formats": ["(3,)f8", "S1", "i4", "S3", "S4", "f8"],
        },
    )
    atoms["xyz"] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    atoms["chain"] = [b"A", b"A"]
    atoms["res_id"] = [1, 2]
    atoms["res_name"] = [b"ALA", b"GLY"]
    atoms["atom_name"] = [b"CA", b"CA"]
    atoms["radius"] = [1.0, 1.0]
    structure = SimpleNamespace(atoms=atoms)

    output = tmp_path / "stat.rmf3"
    writer = ProteinMCRmfWriter(output, structure)
    writer.append(
        atoms["xyz"],
        metadata={
            "Total_Score": 1.25,
            "ProteinMC_Iteration": 7,
            "rmf_file": "stat.rmf3",
        },
    )
    writer.close()

    data = load_rmf_full(output)
    assert data["rmf_frame_series"]["Total_Score"][0] == pytest.approx(1.25)
    assert int(data["rmf_frame_series"]["ProteinMC_Iteration"][0]) == 7


def test_cli_writes_rmf3(tmp_path: Path) -> None:
    _require_pdb2pqr()
    pytest.importorskip("RMF")
    pytest.importorskip("IMP")
    output = tmp_path / "cli.rmf3"
    result = CliRunner().invoke(
        cli,
        [
            str(PDB_148L),
            "--output",
            str(output),
            "--n-iter",
            "2",
            "--n-out",
            "1",
            "--n-written",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert str(output) in result.output
