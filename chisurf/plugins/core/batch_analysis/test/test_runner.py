"""Headless tests for the batch runner (no Qt, no live ChiSurf session)."""

from __future__ import annotations

import csv

from chisurf.plugins.core.batch_analysis.core import runner


# ── fakes ───────────────────────────────────────────────────────────────────
class FakeParam:
    """A stand-in fitting parameter."""

    def __init__(self, name, value, fixed=False):
        self.name = name
        self.value = value
        self.fixed = fixed


class FakeModel:
    """A stand-in model exposing ``parameters_all``."""

    def __init__(self, params):
        self.parameters_all = params


class FakeFit:
    """A stand-in fit exposing a model, chi2r and ``save``."""

    def __init__(self, params, chi2r=1.0):
        self.model = FakeModel(params)
        self.chi2r = chi2r
        self.saved = []

    def save(self, base, fmt, save_curves=False):
        self.saved.append((base, fmt))


class FakeClient:
    """A recording stand-in for the fitting client."""

    def __init__(self, fit):
        self._fit = fit
        self.set_values = []
        self.set_fixed = []

    def get_fit_objects(self):
        return [self._fit]

    def set_parameter_value(self, parameter_name, value, fit_index):
        self.set_values.append((parameter_name, value))
        for p in self._fit.model.parameters_all:
            if p.name == parameter_name:
                p.value = value

    def set_parameter_fixed(self, parameter_name, fixed, fit_index):
        self.set_fixed.append((parameter_name, fixed))
        for p in self._fit.model.parameters_all:
            if p.name == parameter_name:
                p.fixed = fixed


class FakeDataset:
    """A stand-in dataset with a name and experiment."""

    def __init__(self, name, experiment=None):
        self.name = name
        self.experiment = experiment


# ── pure helpers ────────────────────────────────────────────────────────────
def test_sanitize_filename():
    assert runner.sanitize_filename("/a/b/c d.txt") == "c_d"
    assert runner.sanitize_filename("") == "file"


def test_build_queue_orders_datasets_then_files():
    ds = FakeDataset("Sample A")
    items = runner.build_queue([ds], ["/tmp/x.sm", "/tmp/y.sm"])
    assert [i.kind for i in items] == ["dataset", "file", "file"]
    assert items[0].name == "Sample A"
    assert items[1].name == "/tmp/x.sm"


def test_datasets_have_mixed_types():
    class ExpA:
        pass

    class ExpB:
        pass

    assert not runner.datasets_have_mixed_types([FakeDataset("a", ExpA())])
    assert not runner.datasets_have_mixed_types(
        [FakeDataset("a", ExpA()), FakeDataset("b", ExpA())]
    )
    assert runner.datasets_have_mixed_types([FakeDataset("a", ExpA()), FakeDataset("b", ExpB())])


def test_snapshot_and_restore_roundtrip():
    fit = FakeFit([FakeParam("tau", 4.0, False), FakeParam("x", 0.5, True)])
    snap = runner.snapshot_parameters(fit)
    client = FakeClient(fit)
    # mutate, then restore
    fit.model.parameters_all[0].value = 99.0
    runner.restore_parameters(client, 0, fit, snap)
    assert fit.model.parameters_all[0].value == 4.0
    assert ("tau", 4.0) in client.set_values


def test_collect_rows():
    fit = FakeFit([FakeParam("tau", 4.0, False)], chi2r=1.23)
    rows = runner.collect_rows(fit, 2, "file.sm", "key")
    assert rows[0]["Run"] == "2"
    assert rows[0]["Parameter"] == "tau"
    assert rows[0]["Fixed"] == "No"
    assert rows[0]["Chi2r"] == 1.23


def test_write_csv(tmp_path):
    rows = [
        {
            "Run": "1",
            "Filename": "a",
            "GroupKey": "k",
            "Parameter": "tau",
            "Fixed": "No",
            "Value": 4.0,
            "Chi2r": 1.0,
        }
    ]
    out = tmp_path / "r.csv"
    runner.write_csv(rows, str(out))
    read = list(csv.DictReader(out.open()))
    assert read[0]["Parameter"] == "tau"
    assert "GroupKey" not in read[0]  # dropped by FIELDNAMES


# ── orchestration ───────────────────────────────────────────────────────────
def test_run_batch_restores_between_runs_and_collects(tmp_path):
    fit = FakeFit([FakeParam("tau", 4.0, False)])
    client = FakeClient(fit)
    dispatched = []

    def dispatch(name, payload):
        dispatched.append((name, payload))

    items = runner.build_queue([], ["/tmp/a.sm", "/tmp/b.sm"])
    progress = []
    results = runner.run_batch(
        0,
        items,
        fit_client=client,
        dispatch=dispatch,
        imported_datasets=[],
        on_progress=lambda i, n, name: progress.append((i, n)),
        fit_export_dir=str(tmp_path),
    )
    # two files → two params rows
    assert len(results.rows) == 2
    assert results.file_order == ["/tmp/a.sm", "/tmp/b.sm"]
    assert progress == [(1, 2), (2, 2)]
    # parameters restored before each run → 2 restore calls
    assert client.set_values.count(("tau", 4.0)) == 2
    # file items dispatch dataset.add + set_dataset + run
    names = [n for n, _ in dispatched]
    assert names.count("dataset.add") == 2
    assert names.count("fit.run") == 2
    # per-run exports written
    assert len(fit.saved) == 2


def test_run_batch_dataset_items_use_index(tmp_path):
    fit = FakeFit([FakeParam("tau", 4.0)])
    client = FakeClient(fit)
    ds = FakeDataset("Sample")
    dispatched = []
    items = runner.build_queue([ds], [])
    runner.run_batch(
        0,
        items,
        fit_client=client,
        dispatch=lambda name, payload: dispatched.append((name, payload)),
        imported_datasets=[ds],
    )
    set_ds = [p for n, p in dispatched if n == "fit.set_dataset"]
    assert set_ds[0]["dataset_index"] == 0


def test_run_batch_screenshot_callback_recorded():
    fit = FakeFit([FakeParam("tau", 4.0)])
    client = FakeClient(fit)
    items = runner.build_queue([], ["/tmp/a.sm"])
    results = runner.run_batch(
        0,
        items,
        fit_client=client,
        dispatch=lambda **k: None if False else None,
        imported_datasets=[],
        on_run_complete=lambda item, i, key: f"/shots/{i}.png",
    )
    assert list(results.screenshot_map.values()) == ["/shots/1.png"]


def test_run_batch_bad_index_raises():
    fit = FakeFit([FakeParam("tau", 4.0)])
    client = FakeClient(fit)
    try:
        runner.run_batch(
            5,
            runner.build_queue([], ["/tmp/a.sm"]),
            fit_client=client,
            dispatch=lambda **k: None,
            imported_datasets=[],
        )
    except IndexError:
        return
    raise AssertionError("expected IndexError")
