"""GUI tests: declarative AutoForm settings + dockable plots."""

import numpy as np


def test_tool_builds_with_declarative_settings_and_dock_plots(qapp, qtbot):
    from chisurf.gui.autoform.sections.builtin import PlotWidget, ValueWidget
    from chisurf.gui.widgets.dock_area import DockArea
    from chisurf.plugins.burst.burst_fcs_correlator.gui.tool import BurstFcsTool

    w = BurstFcsTool()
    qtbot.addWidget(w)

    # settings rendered declaratively via AutoForm
    attrs = {v._section.attr for v in w._settings_form.findChildren(ValueWidget)}
    assert {"n_bins", "n_casc", "padding_ms"} <= attrs

    # plots live in a declarative dock_area with two plot panels
    assert len(w._plots_form.findChildren(DockArea)) == 1
    assert len(w._plots_form.findChildren(PlotWidget)) == 2

    # selecting a curve drives the plot sources
    tau = np.logspace(-3, 2, 40)
    g = 0.5 / (1.0 + tau) + 1.0
    w._curves = [{
        "file": "f.ptu", "burst_index": 0, "pair_name": "GG",
        "tau_raw": tau.tolist(), "g_raw": g.tolist(),
        "tau": tau.tolist(), "g": g.tolist(), "g_fit": (g * 0.99).tolist(),
        "td_grid": [], "p": [],
    }]
    w._refresh_browser_list()
    assert w.list_browser.count() == 1
    w.list_browser.setCurrentRow(0)
    assert w._model._selected is not None
    assert len(w._model.corr_plot_series()) == 2  # data + fit


def test_settings_model_to_core_settings(qapp):
    from chisurf.plugins.burst.burst_fcs_correlator.gui.tool import _BurstFcsModel

    m = _BurstFcsModel()
    m.maxent_log10_reg = -1.0
    m.fit_mode = "maxent"
    s = m.to_settings()
    assert s.fit_mode == "maxent"
    assert abs(s.maxent_reg - 0.1) < 1e-9  # 10**-1
