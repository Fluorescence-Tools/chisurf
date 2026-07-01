"""Tests for the AutoForm-based TTTR Split / Convert tool."""

from __future__ import annotations

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_PTU = _REPO_ROOT / "test" / "data" / "clsm" / "Leica_SP5.ptu"


def test_view_model_defaults_and_options():
    from chisurf.plugins.tttr.tttr_splitter.gui.view_model import SplitterViewModel

    m = SplitterViewModel()
    assert m.input_format == "Auto"
    assert m.output_format == "PTU"
    assert m.photons_per_file == 300_000
    assert m.split_files and m.reset_macro_times and m.keep_original
    assert m.input_format_options()[0] == "Auto"
    assert "PTU" in m.output_format_options()
    # No file loaded yet → a split is refused with a reason.
    assert m.can_split() is not None


def test_view_spec_loads():
    from chisurf.plugins.tttr.tttr_splitter.gui.view_model import SplitterViewModel

    spec = SplitterViewModel().view_spec()
    assert spec is not None


def test_tool_builds_with_autoform(qapp, qtbot):
    from chisurf.gui.autoform import AutoForm
    from chisurf.plugins.tttr.tttr_splitter.gui.tool import PTUSplitter

    w = PTUSplitter()
    qtbot.addWidget(w)
    assert isinstance(w.auto_form, AutoForm)
    # the custom sections must be registered (imported by gui.tool)
    from chisurf.gui.autoform.sections.registry import get_section_factory

    for key in ("splitter_io", "splitter_run", "splitter_batch_run", "path_list"):
        assert get_section_factory(key) is not None


@pytest.mark.skipif(not _PTU.exists(), reason="sample PTU not available")
def test_split_preserves_photons(tmp_path):
    import tttrlib

    from chisurf.plugins.tttr.tttr_splitter.gui.view_model import SplitterViewModel

    src = str(_PTU)
    n = len(tttrlib.TTTR(src))

    m = SplitterViewModel()
    m.set_tttr(tttrlib.TTTR(src), src)
    m.output_folder = str(tmp_path)
    m.split_files = True
    m.photons_per_file_k = max(1, n // 3000)  # ~3 chunks
    m.output_format = "PTU"

    seen: list[int] = []
    sub = m.do_split(progress_cb=seen.append)
    files = sorted(sub.glob("*.ptu"))

    assert len(files) >= 2
    assert seen and seen[-1] == 100
    total = sum(len(tttrlib.TTTR(str(f))) for f in files)
    assert total == n
