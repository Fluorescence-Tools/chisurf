"""Migration tests: api/core, backend RPC, manifest, CLI, GUI client wiring."""

import pathlib

import numpy as np


def _synthetic(n=64):
    rng = np.random.default_rng(0)
    total = np.abs(rng.random(n)) + 1.0
    sp1 = np.exp(-np.arange(n) / 10.0)
    sp2 = np.exp(-np.arange(n) / 30.0)
    return total, [sp1, sp2]


def test_rpc_client_matches_direct_api():
    from chisurf.plugins.fcs.fcs_filter_calculator import api
    from chisurf.plugins.fcs.fcs_filter_calculator.api import FilterResult
    from chisurf.plugins.fcs.fcs_filter_calculator.gui.client import FilterCalcClient

    total, species = _synthetic()
    direct = api.compute_filters(total, species)
    r = FilterCalcClient().compute(total.tolist(), [s.tolist() for s in species])
    assert r["ok"] is True
    rec = FilterResult.from_dict(r["result"])
    assert np.allclose(rec.filters, direct.filters)


def test_manifest_and_entrypoints():
    from chisurf.core.plugin import load_manifest

    here = pathlib.Path(__file__).resolve().parent.parent
    m = load_manifest(here / "manifest.json")
    assert m is not None and m.id == "fcs_filter_calculator"
    assert m.menu_hidden is True
    assert m.entrypoints.cli and m.entrypoints.services
    names = {rpc.name for rpc in m.rpc_methods}
    assert {"fcs_filter.compute", "fcs_filter.compute_mfd_from_files"} <= names


def test_cli_compute_and_info(tmp_path):
    from chisurf.plugins.fcs.fcs_filter_calculator.cli.main import main

    total, species = _synthetic()
    total_f = tmp_path / "total.txt"
    np.savetxt(total_f, total)
    sp_files = []
    for i, s in enumerate(species):
        f = tmp_path / f"sp{i}.txt"
        np.savetxt(f, s)
        sp_files.append(str(f))
    out = tmp_path / "filters.json"
    args = ["compute", "-t", str(total_f)]
    for f in sp_files:
        args += ["-s", f]
    args += ["-o", str(out)]
    main(args)
    assert out.exists()


def test_widget_routes_through_client(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_filter_calculator.gui import FcsFilterCalculatorWidget

    w = FcsFilterCalculatorWidget()
    qtbot.addWidget(w)
    total, species = _synthetic()
    res = w._compute_filters_rpc(total, species)
    assert res.n_species == 2
