"""Headless tests for the batch-analysis view-model + view spec (no Qt)."""

from __future__ import annotations

from chisurf.core.dataspec import WizardSection, WizardStepSection
from chisurf.plugins.core.batch_analysis.gui.view_model import BatchViewModel


def test_view_spec_is_a_wizard_of_steps():
    spec = BatchViewModel().view_spec()
    assert len(spec.sections) == 1
    wiz = spec.sections[0]
    assert isinstance(wiz, WizardSection)
    assert all(isinstance(s, WizardStepSection) for s in wiz.steps)
    titles = [s.title for s in wiz.steps]
    assert titles[0] == "Welcome"
    assert "Files & fit" in titles
    assert "Results" in titles


def test_every_reference_resolves_on_the_model():
    vm = BatchViewModel()
    wiz = vm.view_spec().sections[0]
    for step in wiz.steps:
        if step.complete_when:
            attr = step.complete_when["attr"]
            assert hasattr(vm, attr), f"missing complete_when attr {attr!r}"
        for sec in step.sections:
            src = getattr(sec, "source", "")
            if src:
                assert callable(getattr(vm, src, None)), f"missing source {src!r}"
            opt_src = getattr(sec, "options_source", "")
            if opt_src:
                assert callable(getattr(vm, opt_src, None)), f"missing options_source {opt_src!r}"
            for btn in getattr(sec, "buttons", ()):
                action = btn.get("action", "")
                assert callable(getattr(vm, action, None)), f"missing action {action!r}"


def test_completion_flags():
    vm = BatchViewModel()
    assert not vm.has_data
    assert not vm.has_results
    vm.files = ["/tmp/a.sm"]
    assert vm.has_data


def test_build_items_from_files():
    vm = BatchViewModel()
    vm.files = ["/tmp/a.sm", "/tmp/b.sm"]
    items = vm.build_items()
    assert [i.kind for i in items] == ["file", "file"]


def test_selected_datasets_maps_indices(monkeypatch):
    vm = BatchViewModel()

    class DS:
        def __init__(self, n):
            self.name = n

    datasets = [DS("a"), DS("b"), DS("c")]
    monkeypatch.setattr(vm, "imported_datasets", lambda: datasets)
    vm.selected_dataset_indices = [0, 2]
    got = [d.name for d in vm.selected_datasets()]
    assert got == ["a", "c"]


def test_info_sources_return_strings():
    vm = BatchViewModel()
    assert vm.welcome_html().strip()
    assert "<" in vm.selection_html()
    assert "<" in vm.results_html()
    assert "<" in vm.status_html()


def test_fit_index_resolution():
    vm = BatchViewModel()
    # no fits available → fit_names() returns [] → fit_index -1
    vm.selected_fit_name = "does-not-exist"
    assert vm.fit_index() == -1
