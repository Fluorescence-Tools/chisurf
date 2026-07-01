"""Tests for the ALEX Creator plugin (core / api / cli / gui split + batch)."""

from __future__ import annotations

import pathlib
import shutil

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_PTU = _REPO_ROOT / "test" / "data" / "clsm" / "Leica_SP5.ptu"


def test_view_model_defaults_and_options():
    from chisurf.plugins.tttr.ptu_alex_creator.gui.view_model import AlexViewModel

    m = AlexViewModel()
    assert m.input_format == "Auto"
    assert m.output_format == "PTU"
    assert m.alex_period == 8000
    assert m.input_format_options()[0] == "Auto"
    assert "PTU" in m.output_format_options()
    assert m.can_save() is not None  # nothing loaded
    assert m.can_run_batch() is not None  # no batch files


def test_view_spec_loads():
    from chisurf.plugins.tttr.ptu_alex_creator.gui.view_model import AlexViewModel

    assert AlexViewModel().view_spec() is not None


def test_manifest_and_cli_and_rpc():
    from pathlib import Path

    from chisurf.core.plugin import load_manifest
    from chisurf.plugins.tttr.ptu_alex_creator.backend.services import list_methods
    from chisurf.plugins.tttr.ptu_alex_creator.cli.main import cli

    m = load_manifest(Path(__file__).resolve().parents[1] / "manifest.json")
    assert m is not None
    assert m.entrypoints.gui and m.entrypoints.cli and m.entrypoints.services
    assert len(m.rpc_methods) == 4
    assert set(cli.commands) == {"convert", "merge"}
    assert "alex.convert" in list_methods() and "alex.merge" in list_methods()


def test_batch_drop_expands_files_and_folders(qapp, qtbot, tmp_path):
    from chisurf.plugins.tttr.ptu_alex_creator.gui.sections import _BatchSection
    from chisurf.plugins.tttr.ptu_alex_creator.gui.view_model import AlexViewModel

    (tmp_path / "m1.sm").write_text("")
    (tmp_path / "m2.sm").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "m3.ptu").write_text("")
    (tmp_path / "notes.txt").write_text("")

    section = _BatchSection(AlexViewModel())
    qtbot.addWidget(section)
    # a folder drop expands recursively to TTTR files only; plain files pass through
    names = sorted(pathlib.Path(p).name for p in section._expand_paths([str(tmp_path)]))
    assert names == ["m1.sm", "m2.sm", "m3.ptu"]
    multi = section._expand_paths([str(tmp_path / "m1.sm"), str(tmp_path / "m2.sm")])
    assert len(multi) == 2


def test_tool_builds_with_autoform(qapp, qtbot):
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.registry import get_section_factory
    from chisurf.plugins.tttr.ptu_alex_creator.gui.tool import AlexPTUCreator

    w = AlexPTUCreator()
    qtbot.addWidget(w)
    assert isinstance(w.auto_form, AutoForm)
    assert get_section_factory("alex_actions") is not None
    assert get_section_factory("alex_batch") is not None


@pytest.mark.skipif(not _PTU.exists(), reason="sample PTU not available")
def test_core_convert_and_histogram(tmp_path):
    import tttrlib

    from chisurf.plugins.tttr.ptu_alex_creator import core

    n = len(tttrlib.TTTR(str(_PTU)))
    out = tmp_path / "single_alex.ptu"
    core.convert_file(str(_PTU), str(out), 8000, 0, "PTU", "Auto")
    assert out.exists() and len(tttrlib.TTTR(str(out))) == n

    hist = core.alex_histogram(str(_PTU), 4000, 0, core.resolve_filetype("Auto", str(_PTU)))
    assert len(hist) == 4000 and int(hist.sum()) == n


@pytest.mark.skipif(not _PTU.exists(), reason="sample PTU not available")
def test_api_batch_convert_and_merge(tmp_path):
    import tttrlib

    from chisurf.plugins.tttr.ptu_alex_creator.api import AlexRequest, run

    n = len(tttrlib.TTTR(str(_PTU)))
    f1 = tmp_path / "a.ptu"
    f2 = tmp_path / "b.ptu"
    shutil.copy(_PTU, f1)
    shutil.copy(_PTU, f2)

    conv = run(
        AlexRequest(
            files=[str(f1), str(f2)],
            output_format="PTU",
            mode="convert",
            output_dir=str(tmp_path / "out"),
        )
    )
    assert len(conv.output_paths) == 2
    assert all(pathlib.Path(p).exists() for p in conv.output_paths)

    merged = run(
        AlexRequest(
            files=[str(f1), str(f2)],
            output_format="PTU",
            mode="merge",
            output_path=str(tmp_path / "m.ptu"),
        )
    )
    assert len(merged.output_paths) == 1
    # merge concatenates every photon from both files
    assert len(tttrlib.TTTR(merged.output_paths[0])) == 2 * n
