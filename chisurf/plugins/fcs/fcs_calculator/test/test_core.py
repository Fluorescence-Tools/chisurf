"""Foundation tests for the migrated FCS confocal calculator (core/backend/RPC)."""

import pathlib


def test_core_compute_confocal_fix_D():
    from chisurf.plugins.fcs.fcs_calculator.core import compute_confocal

    r = compute_confocal(
        tau_us=70.0, S=5.0, temp_C=20.0, eta_mPa_s=0.89, use_water_eta=True,
        constraint="D", D_um2_s=400.0, rh_nm=0.5, veff_fL=0.4,
        conc_nM=1.0, num_mols=0.0, invN=0.0, last_edited="conc",
    )
    assert abs(r["rh_nm"] - 0.5359) < 1e-3
    assert abs(r["veff_fL"] - 1.0436) < 1e-3
    assert r["num_mols"] > 0 and r["invN"] > 0


def test_inprocess_rpc_client_matches_core():
    from chisurf.plugins.fcs.fcs_calculator.core import compute_confocal
    from chisurf.plugins.fcs.fcs_calculator.gui.client import ConfocalCalcClient

    kwargs = dict(
        tau_us=50.0, S=4.0, temp_C=25.0, eta_mPa_s=0.89, use_water_eta=True,
        constraint="rh", D_um2_s=300.0, rh_nm=0.7, veff_fL=0.5,
        conc_nM=2.0, num_mols=0.0, invN=0.0, last_edited="conc",
    )
    direct = compute_confocal(**kwargs)
    r = ConfocalCalcClient().compute(**kwargs)
    assert r["ok"] is True
    assert abs(r["result"]["D_um2_s"] - direct["D_um2_s"]) < 1e-9


def test_manifest_and_menu_hidden():
    from chisurf.core.plugin import load_manifest

    here = pathlib.Path(__file__).resolve().parent.parent
    m = load_manifest(here / "manifest.json")
    assert m is not None and m.id == "fcs_calculator"
    assert m.menu_hidden is True
    names = {rpc.name for rpc in m.rpc_methods}
    assert "fcs_calculator.compute" in names
    assert m.entrypoints.cli  # CLI entrypoint declared


def test_cli_runs():
    from chisurf.plugins.fcs.fcs_calculator.cli.main import main

    assert main(["--constraint", "D", "--tau-us", "70"]) == 0
