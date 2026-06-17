"""Tests for the FRET Calculator plugin manifest and core algorithms."""

from __future__ import annotations

import json
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PLUGIN_DIR / "manifest.json"


class TestManifest:
    """Validate the plugin manifest."""

    def test_manifest_exists(self) -> None:
        assert MANIFEST_PATH.exists(), f"manifest.json not found at {MANIFEST_PATH}"

    def test_manifest_is_valid_json(self) -> None:
        data = json.loads(MANIFEST_PATH.read_text())
        assert isinstance(data, dict)
        assert data["id"] == "fret_calculator"
        assert "version" in data

    def test_manifest_has_required_entrypoints(self) -> None:
        data = json.loads(MANIFEST_PATH.read_text())
        entrypoints = data.get("entrypoints", {})
        assert "gui" in entrypoints
        assert "services" in entrypoints

    def test_manifest_rpc_methods_have_names(self) -> None:
        data = json.loads(MANIFEST_PATH.read_text())
        for method in data.get("rpc_methods", []):
            assert "name" in method
            assert isinstance(method["name"], str)
            assert len(method["name"]) > 0


class TestCoreAlgorithms:
    """Validate the pure computation functions."""

    def test_fret_from_distance(self) -> None:
        from ..core.algorithms import compute_fret_from_distance

        result = compute_fret_from_distance(R=50.0, R0=52.0, tau0=4.0)
        assert 0 < result["E"] < 1
        assert result["tau_DA"] > 0
        assert result["kFRET"] > 0
        assert result["R"] == 50.0

    def test_fret_from_efficiency(self) -> None:
        from ..core.algorithms import compute_fret_from_efficiency

        result = compute_fret_from_efficiency(E=0.5, R0=52.0, tau0=4.0)
        assert abs(result["E"] - 0.5) < 1e-10
        assert result["R"] > 0
        assert result["tau_DA"] > 0

    def test_fret_from_lifetime(self) -> None:
        from ..core.algorithms import compute_fret_from_lifetime

        result = compute_fret_from_lifetime(tau_DA=2.0, R0=52.0, tau0=4.0)
        assert abs(result["E"] - 0.5) < 1e-10
        assert abs(result["tau_DA"] - 2.0) < 1e-10

    def test_fret_from_rate(self) -> None:
        from ..core.algorithms import compute_fret_from_rate

        result = compute_fret_from_rate(kFRET=0.25, R0=52.0, tau0=4.0)
        assert result["R"] > 0
        assert result["E"] > 0

    def test_fret_with_sigma(self) -> None:
        from ..core.algorithms import compute_fret_from_distance

        result = compute_fret_from_distance(R=50.0, R0=52.0, tau0=4.0, sigma=5.0)
        assert 0 < result["E"] < 1
        assert result["sigma"] == 5.0

    def test_homo_fret(self) -> None:
        from ..core.algorithms import compute_homo_fret

        result = compute_homo_fret(t_RM=1.0, rho=2.0, tau0=4.0, R0=52.0)
        assert result["k_homo"] > 0
        assert result["R_DA"] > 0

    def test_homo_fret_backmap(self) -> None:
        from ..core.algorithms import compute_homo_fret, compute_homo_fret_backmap

        fwd = compute_homo_fret(t_RM=1.0, rho=2.0, tau0=4.0, R0=52.0)
        bwd = compute_homo_fret_backmap(
            R_DA=fwd["R_DA"], R0=52.0, tau0=4.0, rho=2.0
        )
        assert abs(bwd["t_RM"] - 1.0) < 1e-6
        assert abs(bwd["k_homo"] - fwd["k_homo"]) < 1e-6

    def test_homo_fret_invalid_inputs(self) -> None:
        from ..core.algorithms import compute_homo_fret

        result = compute_homo_fret(t_RM=-1.0, rho=2.0, tau0=4.0, R0=52.0)
        assert result["k_homo"] == 0.0


class TestAPIModels:
    """Validate API dataclass round-trips."""

    def test_fret_settings_round_trip(self) -> None:
        from ..api.models import FretSettings

        s = FretSettings(R=60.0, R0=54.0, tau0=3.5, sigma=2.0)
        d = s.to_dict()
        s2 = FretSettings.from_dict(d)
        assert s2.R == 60.0
        assert s2.sigma == 2.0

    def test_fret_result_round_trip(self) -> None:
        from ..api.models import FretResult

        r = FretResult(R=50.0, E=0.5, kFRET=0.25)
        d = r.to_dict()
        r2 = FretResult.from_dict(d)
        assert r2.R == 50.0
        assert r2.E == 0.5

    def test_homo_fret_settings_round_trip(self) -> None:
        from ..api.models import HomoFretSettings

        s = HomoFretSettings(t_RM=1.5, rho=3.0)
        d = s.to_dict()
        s2 = HomoFretSettings.from_dict(d)
        assert s2.t_RM == 1.5
        assert s2.rho == 3.0
