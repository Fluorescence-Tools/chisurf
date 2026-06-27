"""Offscreen-Qt tests for the AutoForm-based ``ParameterEditor``.

``ParameterEditor`` renders an arbitrary settings dict through AutoForm (no
pyqtgraph). These tests walk the path its real consumers use — the distribution
plot controller swaps ``_dict`` and calls ``update()``; callers read ``.dict``
and rely on a value-change callback — and assert it builds, mutates in place,
and round-trips non-scalar values. Qt runs offscreen so the tests stay headless.
"""

from __future__ import annotations

import json
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


def _editor(qapp, **kwargs):
    from chisurf.gui.widgets.parameter_editor import ParameterEditor

    return ParameterEditor(json_file="", **kwargs)


def test_builds_and_dict_reflects_the_backing_data(qapp):
    data = {"group": {"value": 1.0}, "flag": True, "name": "x"}
    pe = _editor(qapp, target=data)
    assert pe.dict == data  # snapshot mirrors the backing dict


def test_dict_normalizes_none_string_for_pyqtgraph(qapp):
    # Regression: a curve option symbol="None" must come back as Python None,
    # else pyqtgraph's ScatterPlotItem raises KeyError: 'None'.
    data = {"curve_options": {"symbol": "None", "label": "ok"}, "top": "None"}
    pe = _editor(qapp, target=data)
    out = pe.dict
    assert out["curve_options"]["symbol"] is None
    assert out["top"] is None
    assert out["curve_options"]["label"] == "ok"
    # the live backing dict is left untouched (still the literal string)
    assert data["curve_options"]["symbol"] == "None"


def test_value_callback_fires_on_edit(qapp):
    from chisurf.core.dataspec import SettingsView

    data = {"threshold": 5}
    fired = []
    pe = _editor(qapp, target=data, callback=lambda: fired.append(1))
    # Drive a change the way AutoForm would: resolve the group, set the attr.
    view = SettingsView(data, on_change=pe._on_change)
    sec = view.view_spec().sections[0]
    setattr(getattr(view, sec.target), sec.attr, 9)
    assert fired == [1]


def test_dict_swap_then_update_rebuilds(qapp):
    # The distribution-plot controller does exactly this.
    pe = _editor(qapp, target={"a": 1})
    pe._dict = {"b": {"c": 2.0}, "d": "hi"}
    pe.update()
    assert pe.dict == {"b": {"c": 2.0}, "d": "hi"}


def test_callable_value_survives_round_trip(qapp):
    fn = lambda x: x  # noqa: E731
    data = {"opts": {"accessor": fn, "symbol": "o"}}
    pe = _editor(qapp, target=data)
    assert pe.dict["opts"]["accessor"] is fn


def _leaf_items(pe):
    """All leaf tree items (those carrying a value editor in column 1), by label."""
    out = {}

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            if pe._tree.itemWidget(child, 1) is not None:
                out[child.text(0)] = child
            walk(child)

    walk(pe._tree.invisibleRootItem())
    return out


def test_editing_inline_spinbox_mutates_dict_and_fires_callback(qapp):
    data = {"threshold": 5, "scale": 1.0}
    fired = []
    pe = _editor(qapp, target=data, callback=lambda: fired.append(1))
    items = _leaf_items(pe)
    spin = pe._tree.itemWidget(items["threshold"], 1)  # QSpinBox
    spin.setValue(11)
    assert data["threshold"] == 11
    assert fired  # callback fired on edit


def test_reset_button_restores_default(qapp):
    data = {"alpha": 2.0}
    pe = _editor(qapp, target=data)
    item = _leaf_items(pe)["alpha"]
    editor = pe._tree.itemWidget(item, 1)
    reset_btn = pe._tree.itemWidget(item, 2)
    editor.setValue(9.0)
    assert data["alpha"] == 9.0
    reset_btn.click()
    assert data["alpha"] == 2.0  # back to the build-time default


def test_readonly_value_has_no_reset_button(qapp):
    fn = lambda x: x  # noqa: E731
    data = {"accessor": fn, "label": "ok"}
    pe = _editor(qapp, target=data)
    items = _leaf_items(pe)
    assert pe._tree.itemWidget(items["accessor"], 2) is None  # no reset for callable
    assert pe._tree.itemWidget(items["label"], 2) is not None  # editable str has one


def test_filter_hides_non_matching_rows(qapp):
    data = {"alpha": 1, "beta": 2, "alphabet": 3}
    pe = _editor(qapp, target=data)
    items = _leaf_items(pe)
    pe._filter.setText("alpha")
    assert items["alpha"].isHidden() is False
    assert items["alphabet"].isHidden() is False
    assert items["beta"].isHidden() is True
    pe._filter.setText("")  # cleared -> all visible again
    assert items["beta"].isHidden() is False


def test_filter_keeps_group_when_child_matches(qapp):
    data = {"group": {"target_param": 1, "other": 2}, "top": 3}
    pe = _editor(qapp, target=data)
    pe._filter.setText("target")
    items = _leaf_items(pe)
    assert items["target_param"].isHidden() is False
    assert items["target_param"].parent().isHidden() is False  # group stays visible
    assert items["top"].isHidden() is True


def test_list_value_renders_combobox_and_moves_choice_to_front(qapp):
    from qtpy import QtWidgets

    data = {"polarization_options": ["vm", "vv", "vh"]}
    pe = _editor(qapp, target=data)
    item = _leaf_items(pe)["polarization_options"]
    combo = pe._tree.itemWidget(item, 1)
    assert isinstance(combo, QtWidgets.QComboBox)
    assert [combo.itemText(i) for i in range(combo.count())] == ["vm", "vv", "vh"]
    combo.setCurrentIndex(2)  # pick 'vh'
    # selection moves to front; the full list is preserved (reversible)
    assert data["polarization_options"] == ["vh", "vm", "vv"]


def test_choice_section_renders_combobox_and_updates(qapp):
    from qtpy import QtWidgets
    from chisurf.core.dataspec import ChoiceSection, ModelView

    data = {"covariance_type": "full"}
    pe = _editor(qapp, target=data)
    
    class DummyGroup:
        def __init__(self, d):
            self._d = d
            self.covariance_type = d["covariance_type"]
        def __setattr__(self, name, val):
            if name == "_d":
                super().__setattr__(name, val)
            else:
                self._d[name] = val
                super().__setattr__(name, val)

    pe._view.settings = DummyGroup(data)
    view_spec = ModelView(sections=(
        ChoiceSection(target="settings", attr="covariance_type",
                      label="covariance_type",
                      options=("full", "tied", "diag", "spherical")),
    ))

    pe._tree.clear()
    pe._add_sections(view_spec.sections, pe._tree.invisibleRootItem())

    item = _leaf_items(pe)["covariance_type"]
    combo = pe._tree.itemWidget(item, 1)
    assert isinstance(combo, QtWidgets.QComboBox)
    assert [combo.itemText(i) for i in range(combo.count())] == ["full", "tied", "diag", "spherical"]
    assert combo.currentText() == "full"

    combo.setCurrentIndex(2)  # "diag"
    assert data["covariance_type"] == "diag"


def test_save_writes_json(tmp_path, qapp):
    from chisurf.gui.widgets.parameter_editor import ParameterEditor

    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"alpha": 1, "beta": "two"}))
    pe = ParameterEditor(json_file=str(f), target={})
    pe._dict["alpha"] = 7  # widgets mutate the live backing dict; save() writes it
    pe.save()
    assert json.loads(f.read_text())["alpha"] == 7
