"""Headless tests for the anisotropy view-model + view spec (no Qt)."""

from __future__ import annotations

from chisurf.core.dataspec import WizardSection, WizardStepSection
from chisurf.plugins.fluorescence_decay.tr_anisotropy.gui.view_model import AnisotropyViewModel


def test_view_spec_is_a_wizard_of_steps():
    spec = AnisotropyViewModel().view_spec()
    assert len(spec.sections) == 1
    wiz = spec.sections[0]
    assert isinstance(wiz, WizardSection)
    assert all(isinstance(s, WizardStepSection) for s in wiz.steps)
    titles = [s.title for s in wiz.steps]
    assert titles[0] == "Welcome"
    assert "Normalize IRF" in titles
    assert "Components" in titles
    assert "Finish" in titles


def test_every_reference_resolves_on_the_model():
    vm = AnisotropyViewModel()
    wiz = vm.view_spec().sections[0]
    for step in wiz.steps:
        if step.complete_when:
            attr = step.complete_when["attr"]
            assert hasattr(vm, attr), f"missing complete_when attr {attr!r}"
        for sec in step.sections:
            src = getattr(sec, "source", "")
            if src:
                assert callable(getattr(vm, src, None)), f"missing source {src!r}"
            attr = getattr(sec, "attr", None)
            if attr:
                assert hasattr(vm, attr), f"missing value/choice attr {attr!r}"
            for btn in getattr(sec, "buttons", ()):
                action = btn.get("action", "")
                assert callable(getattr(vm, action, None)), f"missing action {action!r}"


def test_embedded_widgets_declared():
    wiz = AnisotropyViewModel().view_spec().sections[0]
    embeds = [
        sec for step in wiz.steps for sec in step.sections if getattr(sec, "key", "") == "embed"
    ]
    widgets = {sec.options.get("widget", "") for sec in embeds}
    assert any("IrfNormalizationWidget" in w for w in widgets)
    assert any("ComponentsWidget" in w for w in widgets)


def test_corrections_are_float_properties():
    vm = AnisotropyViewModel()
    vm.g_factor = 1.5
    vm.l1 = 0.11
    vm.l2 = 0.22
    assert vm.g_factor == 1.5
    assert vm._corrections["l1"] == 0.11


def test_apply_region_without_data_is_safe():
    vm = AnisotropyViewModel()
    # no curves loaded → should not raise, corrected IRFs stay None
    vm.apply_region(10, 20)
    assert vm.data["irf_vv_bg_norm"] is None


def test_components_ready_flag():
    vm = AnisotropyViewModel()
    assert vm.components_ready  # defaults have components
    vm.lifetime_spectrum = []
    assert not vm.components_ready


def test_info_sources_return_strings():
    vm = AnisotropyViewModel()
    assert "<" in vm.welcome_html()
    assert "<" in vm.data_html()
    assert "<" in vm.finish_html()


def test_plot_series_empty_without_data():
    assert AnisotropyViewModel().plot_series() == []
