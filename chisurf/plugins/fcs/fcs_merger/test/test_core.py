"""Migration tests for the FCS-Merger plugin (core / RPC / manifest / CLI)."""

import pathlib

import numpy as np


def _corr(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 32
    x = np.logspace(-3, 2, n)
    y = 0.5 / (1.0 + x) + 1.0 + rng.normal(0, 0.01, n)
    return {
        "x": x.tolist(), "y": y.tolist(), "duration": 10.0, "count_rate": 50.0,
        "channel_a": {"counts": 5e5}, "channel_b": {"counts": 5e5},
    }


def test_core_average_and_merge_folder(tmp_path):
    from chisurf.plugins.fcs.fcs_merger.core import (
        compute_average_correlations,
        merge_folder,
        save_mean_correlation,
    )

    m = compute_average_correlations([_corr(1), _corr(2)])
    assert m["duration"] == 20.0
    assert np.any(np.asarray(m["ey"]) != 0)  # std error for >1 curve

    save_mean_correlation(m, tmp_path / "a.cor")
    res = merge_folder(tmp_path, tmp_path / "out.cor")
    assert res["n_curves"] == 1
    assert (tmp_path / "out.cor").exists()


def test_inprocess_rpc_client(tmp_path):
    from chisurf.plugins.fcs.fcs_merger.gui.client import FcsMergerClient

    client = FcsMergerClient()
    r = client.average([_corr(1), _corr(2)])
    assert r["ok"] is True
    assert r["result"]["duration"] == 20.0


def test_manifest_and_entrypoints():
    from chisurf.core.plugin import load_manifest

    here = pathlib.Path(__file__).resolve().parent.parent
    m = load_manifest(here / "manifest.json")
    assert m is not None and m.id == "fcs_merger"
    assert m.menu_hidden is True
    assert m.entrypoints.cli and m.entrypoints.services
    names = {rpc.name for rpc in m.rpc_methods}
    assert "fcs_merger.merge_folder" in names


def test_cli(tmp_path, capsys):
    from chisurf.plugins.fcs.fcs_merger.cli.main import main
    from chisurf.plugins.fcs.fcs_merger.core import (
        compute_average_correlations,
        save_mean_correlation,
    )

    save_mean_correlation(compute_average_correlations([_corr(1), _corr(2)]), tmp_path / "c.cor")
    assert main([str(tmp_path), "--summary"]) == 0
    assert "merged" in capsys.readouterr().out
