"""Tests for the AutoForm-backed entity detail form (EntityForm)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy import QtWidgets
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"qtpy unavailable: {exc}")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _specs():
    from mfdb.admin.gui.entity_schema import FieldSpec

    return [
        FieldSpec(name="sample_id", label="Sample ID", widget="str", readonly=True),
        FieldSpec(name="description", label="Description", widget="str"),
        FieldSpec(name="num_of_probes", label="# Probes", widget="int"),
        FieldSpec(name="conc", label="Conc", widget="float"),
        FieldSpec(name="solvent_phase", label="Phase", widget="choice",
                  choices=["liquid", "vitrified", "other"]),
        FieldSpec(name="details", label="Details", widget="text"),
        FieldSpec(name="is_public", label="Public", widget="bool"),
        FieldSpec(name="condition_id", label="Condition", widget="str",
                  fk_target="condition"),
    ]


def test_all_field_kinds_build(qapp):
    from chisurf.gui.autoform.sections.builtin import (
        ChoiceWidget, ToggleWidget, ValueWidget,
    )
    from mfdb.admin.gui.autoform_entity_form import EntityForm

    fk = {"condition_id": lambda: [("c1", "c1 — A"), ("c2", "c2 — B")]}
    f = EntityForm(_specs(), dropdown_providers=fk)
    # 5 ValueSection widgets (str, str, int, float, text), 2 ChoiceWidgets
    # (choice + FK), 1 ToggleWidget — no section failed to build.
    assert len(f.findChildren(ValueWidget)) == 5
    assert len(f.findChildren(ChoiceWidget)) == 2
    assert len(f.findChildren(ToggleWidget)) == 1


def test_set_get_roundtrip(qapp):
    from mfdb.admin.gui.autoform_entity_form import EntityForm

    fk = {"condition_id": lambda: [("c1", "c1 — A"), ("c2", "c2 — B")]}
    f = EntityForm(_specs(), dropdown_providers=fk)
    f.set_data({
        "sample_id": "s1", "description": "my sample", "num_of_probes": 2,
        "conc": 1.5, "solvent_phase": "vitrified", "details": "line1\nline2",
        "is_public": 1, "condition_id": "c2",
    })
    out = f.get_data()
    assert out["sample_id"] == "s1"
    assert out["description"] == "my sample"
    assert out["num_of_probes"] == 2 and isinstance(out["num_of_probes"], int)
    assert out["conc"] == 1.5
    assert out["solvent_phase"] == "vitrified"
    assert out["details"] == "line1\nline2"
    assert out["is_public"] == 1
    assert out["condition_id"] == "c2"  # FK commits the id, not the label


def test_commit_signal_not_fired_during_load(qapp):
    from mfdb.admin.gui.autoform_entity_form import EntityForm

    f = EntityForm(_specs())
    fired = []
    f.commitRequested.connect(lambda: fired.append(1))
    f.set_data({"description": "x", "num_of_probes": 1})
    assert fired == []
    # a user edit (setattr on a field) fires the commit signal
    f._model.description = "edited"
    assert len(fired) == 1


def test_json_fields_roundtrip(qapp):
    from mfdb.admin.gui.entity_schema import FieldSpec
    from mfdb.admin.gui.autoform_entity_form import EntityForm

    specs = [
        FieldSpec(name="laser_wavelengths", widget="text"),
        FieldSpec(name="detector_channels", widget="text"),
    ]
    f = EntityForm(specs)
    f.set_data({"laser_wavelengths": [488, 640], "detector_channels": {"0": "green"}})
    out = f.get_data()
    assert out["laser_wavelengths"] == [488, 640]
    assert out["detector_channels"] == {"0": "green"}


def test_empty_optional_strings_become_none(qapp):
    from mfdb.admin.gui.autoform_entity_form import EntityForm

    f = EntityForm(_specs())
    f.set_data({})  # nothing set
    out = f.get_data()
    assert out["description"] is None
    assert out["num_of_probes"] == 0
    assert out["is_public"] == 0
