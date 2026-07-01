"""Tests for the AutoForm-based PTU Header Editor tool."""

from __future__ import annotations

import json
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_PTU = _REPO_ROOT / "test" / "data" / "clsm" / "Leica_SP5.ptu"


def test_view_model_sample_and_json():
    from chisurf.plugins.tttr.ptu_header_edit.gui.view_model import HeaderEditorViewModel

    m = HeaderEditorViewModel()
    assert len(m.tags) == 6  # sample header
    assert m.json_text.strip().startswith("{")
    assert m.can_save() is not None  # no source file open


def test_view_spec_loads():
    from chisurf.plugins.tttr.ptu_header_edit.gui.view_model import HeaderEditorViewModel

    assert HeaderEditorViewModel().view_spec() is not None


def test_tool_builds_with_autoform(qapp, qtbot):
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.registry import get_section_factory
    from chisurf.plugins.tttr.ptu_header_edit.gui.tool import TagsEditor

    w = TagsEditor()
    qtbot.addWidget(w)
    assert isinstance(w.auto_form, AutoForm)
    assert get_section_factory("header_table") is not None


@pytest.mark.skipif(not _PTU.exists(), reason="sample PTU not available")
def test_header_roundtrip_preserves_tags_and_photons(tmp_path):
    import tttrlib

    from chisurf.plugins.tttr.ptu_header_edit.gui.view_model import HeaderEditorViewModel

    n = len(tttrlib.TTTR(str(_PTU)))
    m = HeaderEditorViewModel()
    m.load_ptu(str(_PTU))
    n_tags = len(m.tags)
    assert n_tags > 10

    # Rebuild rows exactly as the table would; round-trip must not drop tags.
    rows = [
        {
            "name": t["name"],
            "type": m.TYPE_MAPPING.get(t["type"], ""),
            "value": t["value"],
            "idx": str(t.get("idx", -1)),
        }
        for t in m.tags
    ]
    m.set_tags(rows)
    assert len(m.tags) == n_tags  # no silent drops

    out = tmp_path / "edited.ptu"
    m.save(str(out))
    back = tttrlib.TTTR(str(out))
    assert len(back) == n
    assert len(json.loads(back.header.json)["tags"]) == n_tags
