"""Headless tests for the onboarding view-model and its authored view spec.

These assert the wizard's structure and that every ``source``/``action``/
``complete_when`` referenced in ``boarding.view.json`` resolves on the model —
without importing Qt.
"""

from __future__ import annotations

from chisurf.core.dataspec import InfoSection, WizardSection, WizardStepSection
from chisurf.plugins.core.boarding.view_model import BoardingViewModel


def test_view_spec_is_a_wizard_of_steps() -> None:
    """The spec is a single wizard whose entries are all wizard steps."""
    spec = BoardingViewModel().view_spec()
    assert len(spec.sections) == 1
    wiz = spec.sections[0]
    assert isinstance(wiz, WizardSection)
    assert len(wiz.steps) == 8
    assert all(isinstance(s, WizardStepSection) for s in wiz.steps)
    titles = [s.title for s in wiz.steps]
    assert titles[0] == "Welcome"
    assert "FCS channels" in titles
    assert "Experiments" in titles


def test_every_reference_resolves_on_the_model() -> None:
    """Each info source, button action and completion attr exists on the model."""
    vm = BoardingViewModel()
    wiz = vm.view_spec().sections[0]
    for step in wiz.steps:
        if step.complete_when:
            attr = step.complete_when["attr"]
            assert hasattr(vm, attr), f"missing complete_when attr {attr!r}"
        for sec in step.sections:
            src = getattr(sec, "source", "")
            if src:
                assert callable(getattr(vm, src, None)), f"missing source {src!r}"
            for btn in getattr(sec, "buttons", ()):
                action = btn.get("action", "")
                assert callable(getattr(vm, action, None)), f"missing action {action!r}"


def test_info_sources_return_strings() -> None:
    """The HTML info-source methods return non-empty strings."""
    vm = BoardingViewModel()
    assert vm.welcome_html().strip()
    assert "<" in vm.status_html()
    assert "<" in vm.deps_html()


def test_editors_are_embedded_via_custom_sections() -> None:
    """The detector and FCS steps embed existing widgets through the ``embed`` key."""
    wiz = BoardingViewModel().view_spec().sections[0]
    embeds = [
        sec for step in wiz.steps for sec in step.sections if getattr(sec, "key", "") == "embed"
    ]
    widgets = {sec.options.get("widget", "") for sec in embeds}
    assert any("FCSChannelWidget" in w for w in widgets)
    assert any("DetectorWizardPage" in w for w in widgets)


def test_repair_status_updates_after_action(monkeypatch) -> None:
    """A create/restore action records an HTML status without touching the disk."""
    vm = BoardingViewModel()
    monkeypatch.setattr(
        "chisurf.plugins.core.boarding.view_model.utils.copy_defaults",
        lambda overwrite: (True, "done"),
    )
    assert vm.repair_status_html() == ""
    vm.create_missing()
    assert "done" in vm.repair_status_html()


def test_status_html_is_mfdb_aware_for_detectors(monkeypatch) -> None:
    """A connected MFDB with detector setups reports OK, not a MISSING JSON file."""
    from chisurf.plugins.core.boarding import utils

    monkeypatch.setattr(utils, "mfdb_info", lambda: {"connected": True, "path": "/tmp/x.db"})
    monkeypatch.setattr(
        utils,
        "detector_setups_summary",
        lambda: {"count": 2, "store": "mfdb", "detail": "in MFDB"},
    )
    monkeypatch.setattr(
        utils,
        "fcs_setups_summary",
        lambda: {"count": 0, "store": "mfdb", "detail": "in MFDB"},
    )
    html = utils.build_status_html()
    # the misleading raw-file row is gone; MFDB-backed setups reported logically
    assert "detector_setups.json" not in html
    assert "Detector setups" in html
    assert "2 setups in MFDB" in html
    assert "MISSING" not in html.split("Detector setups")[1].split("</tr>")[0]
    # an empty MFDB store is neutral, never "MISSING"
    assert "none yet" in html


def test_setups_row_states() -> None:
    """The setup-row renderer maps count/store to OK / neutral, never a false error."""
    from chisurf.plugins.core.boarding import utils

    assert "MISSING" not in utils._setups_row("D", {"count": 0, "store": "mfdb"})
    assert "OK" in utils._setups_row("D", {"count": 1, "store": "mfdb", "detail": "in MFDB"})
    assert "—" in utils._setups_row("D", {"count": None, "store": "unknown"})


def test_complete_when_step_has_info_child() -> None:
    """Sanity: at least one step carries an InfoSection body."""
    wiz = BoardingViewModel().view_spec().sections[0]
    assert any(isinstance(sec, InfoSection) for step in wiz.steps for sec in step.sections)
