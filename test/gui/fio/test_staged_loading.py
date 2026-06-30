"""Offscreen-Qt tests for the staged-loading GUI layer.

Covers :func:`chisurf.gui.widgets.staged_loading.load_with_progress` (success
and cancel paths) and that the data-loading settings AutoForm renders its
fields. Qt runs offscreen so the tests stay headless.
"""

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
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        from chisurf.gui import initialize_gui_executors

        initialize_gui_executors()
    except Exception:
        pass
    return app


def test_load_with_progress_returns_loader_result(qapp, monkeypatch):
    from chisurf.core.fio import staging
    from chisurf.gui.widgets import staged_loading

    # Fake a slow source: emit a couple of progress events, no real file IO.
    def fake_stage(src, *, progress_cb=None, cancel_cb=None, **kw):
        if progress_cb:
            progress_cb(1_000_000, 2_000_000, 12.3, 0.1)
            progress_cb(2_000_000, 2_000_000, 15.0, 0.0)
        return (staging.Path(src), False)

    monkeypatch.setattr(staging, "stage_path_if_slow", fake_stage)
    monkeypatch.setattr(staged_loading.staging, "stage_path_if_slow", fake_stage)

    result = staged_loading.load_with_progress(
        None, lambda p: f"loaded:{p}", "/some/file.ptu", title="Loading"
    )
    assert result == "loaded:/some/file.ptu"


def test_load_with_progress_cancel_returns_none(qapp, monkeypatch):
    from chisurf.core.fio import staging
    from chisurf.gui.widgets import staged_loading

    def fake_stage(src, *, progress_cb=None, cancel_cb=None, **kw):
        raise staging.StagingCancelled()

    monkeypatch.setattr(staged_loading.staging, "stage_path_if_slow", fake_stage)

    called = {"loader": False}

    def loader(p):
        called["loader"] = True
        return "should-not-happen"

    result = staged_loading.load_with_progress(None, loader, "/x.ptu")
    assert result is None
    assert called["loader"] is False


def test_load_with_progress_propagates_error(qapp, monkeypatch):
    from chisurf.core.fio import staging
    from chisurf.gui.widgets import staged_loading

    # No staging, so the loader runs and its error must propagate.
    monkeypatch.setattr(
        staged_loading.staging,
        "stage_path_if_slow",
        lambda src, **kw: (staging.Path(src), False),
    )

    def boom(p):
        raise RuntimeError("read failed")

    with pytest.raises(RuntimeError, match="read failed"):
        staged_loading.load_with_progress(None, boom, "/x.ptu")


def test_settings_form_renders_fields(qapp):
    from qtpy import QtWidgets

    from chisurf.gui.widgets.staged_loading_settings import DataLoadingSettingsWidget

    w = DataLoadingSettingsWidget()
    spins = w.findChildren(QtWidgets.QDoubleSpinBox)
    checks = w.findChildren(QtWidgets.QCheckBox)
    assert len(spins) == 4  # threshold, min size, chunk, probe
    assert len(checks) == 1  # enabled
    # units + tooltips were carried from the JSON spec
    assert any(s.suffix().strip() == "MB/s" for s in spins)
    assert all(c.toolTip() for c in checks)


def test_settings_model_setter_persists(qapp, monkeypatch):
    from chisurf.gui.widgets import staged_loading_settings as sls

    saved = {}
    monkeypatch.setattr(sls, "set_data_loading_settings", lambda d: saved.update(d))

    model = sls.DataLoadingSettingsModel()
    model.threshold_mbps = 55.0
    model.min_size_mb = 16.0
    assert saved["threshold_mbps"] == 55.0
    assert saved["min_size"] == 16 * 1024 * 1024
