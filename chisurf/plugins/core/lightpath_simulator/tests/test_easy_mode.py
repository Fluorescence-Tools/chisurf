"""Tests for the easy-mode graph builder and preset I/O."""


import numpy as np

from chisurf.plugins.core.lightpath_simulator.backend.crosstalk import WAVELENGTHS
from chisurf.plugins.core.lightpath_simulator.backend.simulator import (
    OpticalPathSimulator,
)
from chisurf.plugins.core.lightpath_simulator.gui.easy_mode import (
    TEMPLATE_LIBRARY_DIR,
    _graph_to_config,
    build_easy_graph,
    load_easy_preset,
    load_last_config,
    load_template_library,
    normalize_lightpath_graph,
    save_easy_preset,
    save_last_config,
)


def _make_mock_db(probe_names: dict | None = None):
    """Create a mock MFDatabase with reasonable default spectra.

    Parameters
    ----------
    probe_names : dict, optional
        Mapping of probe_id -> chromophore_name. When a probe_id is looked
        up via ``get_probe_by_id`` the corresponding name is returned.
        Falls back to "Test Dye" for unknown ids.
    """

    class _MockDB:
        """Fake MFDatabase that returns realistic synthetic spectra."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_probe_by_id(self, pid):
            names = probe_names or {}
            return {"chromophore_name": names.get(pid, "Test Dye")}

        def get_probe_spectrum(self, probe_id, spectrum_type):
            wl = WAVELENGTHS
            vals = np.ones_like(wl, dtype=float)
            if spectrum_type == "transmission":
                # Dichroic-like: reflect below 500 nm, transmit above
                vals[wl < 500] = 0.0
            elif spectrum_type == "absorption":
                # Dye-like: Gaussian peaked at 520 nm
                vals = np.exp(-0.5 * ((wl - 520) / 25) ** 2)
            elif spectrum_type == "emission":
                # Dye-like: Gaussian peaked at 580 nm
                vals = np.exp(-0.5 * ((wl - 580) / 30) ** 2)
            return wl, vals

        def get_standardized_optical_properties(self, *a, **kw):
            return {"qy": 0.8, "ext_coeff": 92000}

        def close(self):
            pass

    return _MockDB()


class TestBuildEasyGraph:
    """Tests for build_easy_graph()."""

    def test_minimal_config(self):
        """A config with only defaults should produce a valid graph."""
        graph = build_easy_graph({})
        assert "nodes" in graph
        assert "edges" in graph
        assert graph["version"] == 1
        # Should have: light_source, sample, dichroic, forster_radius
        assert len(graph["nodes"]) >= 4

    def test_with_dyes_and_detector(self):
        """Full config produces all expected node types."""
        config = {
            "lasers": "488:1.0, 640:1.0",
            "excitation_dichroic_probe_id": 1,
            "emission_splitter_type": "Dichroic",
            "emission_splitter_probe_id": 2,
            "detectors": [
                {"name": "Green", "bandpass_probe_id": 3, "qe_probe_id": 4},
                {"name": "Red", "bandpass_probe_id": 5, "qe_probe_id": 6},
            ],
            "dyes": {"10": {"qy": 0.8, "ec": 90000}, "11": {"qy": 0.6, "ec": 150000}},
            "kappa2": 0.6667,
            "n": 1.33,
        }
        graph = build_easy_graph(config)
        types = {n["type"] for n in graph["nodes"]}
        assert "light_source" in types
        assert "sample" in types
        assert "splitter" in types
        assert "filter" in types
        assert "detector" in types
        assert "forster_radius" in types
        # 2 splitters: one excitation dichroic and one emission splitter
        splitter_count = sum(1 for n in graph["nodes"] if n["type"] == "splitter")
        assert splitter_count == 2
        # Always 2 detector nodes
        detector_count = sum(1 for n in graph["nodes"] if n["type"] == "detector")
        assert detector_count == 2

    def test_graph_loads_into_simulator(self):
        """Graph produced by build_easy_graph should load into OpticalPathSimulator."""
        config = {
            "lasers": "488:1.0",
            "excitation_dichroic_probe_id": 1,
            "emission_splitter_type": "Dichroic",
            "emission_splitter_probe_id": 2,
            "detectors": [
                {"name": "Main", "bandpass_probe_id": 3, "qe_probe_id": 4},
                {"name": "Ref", "qe_probe_id": 5},
            ],
            "dyes": {"10": {"qy": 0.8, "ec": 90000}},
            "kappa2": 0.6667,
            "n": 1.33,
        }
        graph = build_easy_graph(config)
        db = _make_mock_db()
        sim = OpticalPathSimulator(db)
        sim.load_from_dict(graph)
        states = sim.propagate()
        assert len(states) > 0
        signals = sim.get_detector_signals()
        assert len(signals) > 0

    def test_crosstalk_matrices(self):
        """All three crosstalk matrices are populated after propagation."""
        config = {
            "lasers": "488:1.0, 640:1.0",
            "detectors": [
                {"name": "Green", "bandpass_probe_id": 2, "qe_probe_id": 3},
                {"name": "Red", "qe_probe_id": 4},
            ],
            "dyes": {"10": {"qy": 0.8, "ec": 90000}},
            "kappa2": 0.6667,
            "n": 1.33,
        }
        graph = build_easy_graph(config)
        db = _make_mock_db()
        sim = OpticalPathSimulator(db)
        sim.load_from_dict(graph)
        sim.propagate()
        matrices = sim.get_crosstalk_matrices()
        assert "excitation" in matrices
        assert "emission" in matrices
        assert "detected" in matrices

    def test_forster_radius_results(self):
        """The forster_radius node should contain R0 results after propagation."""
        config = {
            "lasers": "488:1.0",
            "detectors": [
                {"name": "Det", "qe_probe_id": 3},
                {"name": "Ref", "qe_probe_id": 4},
            ],
            "dyes": {"10": {"qy": 0.8, "ec": 90000}, "11": {"qy": 0.6, "ec": 150000}},
            "kappa2": 0.6667,
            "n": 1.33,
        }
        graph = build_easy_graph(config)
        db = _make_mock_db(probe_names={10: "DonorDye", 11: "AcceptorDye"})
        sim = OpticalPathSimulator(db)
        sim.load_from_dict(graph)
        states = sim.propagate()
        forster_results = None
        for ns in states.values():
            results = ns.config.get("_last_results", [])
            if results:
                forster_results = results
                break
        assert forster_results is not None
        # Should have all dye pairs (2 donors × 2 acceptors = 4)
        assert len(forster_results) == 4

    def test_empty_dyes(self):
        """Config with no dyes should still produce a loadable graph."""
        config = {
            "lasers": "488:1.0",
            "detectors": [
                {"name": "Det", "qe_probe_id": 3},
            ],
            "dyes": {},
        }
        graph = build_easy_graph(config)
        db = _make_mock_db()
        sim = OpticalPathSimulator(db)
        sim.load_from_dict(graph)
        states = sim.propagate()
        assert len(states) > 0

    def test_no_detector_config_provided(self):
        """When only one detector is in the config, we get that detector alone."""
        config = {
            "lasers": "488:1.0",
            "detectors": [{"name": "Main", "qe_probe_id": 3}],
            "dyes": {"10": {"qy": 0.8, "ec": 90000}},
        }
        graph = build_easy_graph(config)
        detector_nodes = [n for n in graph["nodes"] if n["type"] == "detector"]
        assert len(detector_nodes) == 1
        assert detector_nodes[0]["title"] == "Main"

    def test_emission_splitters_cascade(self):
        """Multiple emission splitters cascade to N+1 detectors."""
        config = {
            "lasers": "488:1.0, 561:1.0, 640:1.0",
            "excitation_dichroic_probe_id": 1,
            "emission_splitters": [
                {"type": "Dichroic", "probe_id": 2},
                {"type": "Dichroic", "probe_id": 3},
            ],
            "detectors": [
                {"name": "Ch1", "bandpass_probe_id": 4, "qe_probe_id": 5},
                {"name": "Ch2", "bandpass_probe_id": 6, "qe_probe_id": 7},
                {"name": "Ch3", "bandpass_probe_id": 8, "qe_probe_id": 9},
            ],
            "dyes": {"10": {"qy": 0.8, "ec": 90000}, "11": {"qy": 0.5, "ec": 150000}},
            "kappa2": 0.6667,
            "n": 1.33,
        }
        graph = build_easy_graph(config)
        detector_nodes = [n for n in graph["nodes"] if n["type"] == "detector"]
        splitter_nodes = [n for n in graph["nodes"] if n["type"] == "splitter"]
        assert len(detector_nodes) == 3
        assert len(splitter_nodes) == 3  # excitation dichroic + 2 emission splitters
        assert detector_nodes[0]["title"] == "Ch1"
        assert detector_nodes[1]["title"] == "Ch2"
        assert detector_nodes[2]["title"] == "Ch3"
        # Verify splitter types stored in config
        splitter_types = [n["config"].get("splitter_type") for n in splitter_nodes
                          if "splitter_type" in n.get("config", {})]
        assert splitter_types == ["Dichroic", "Dichroic"]

    def test_polarizer_splitter(self):
        """A polarizer-type splitter propagates through the cascade."""
        config = {
            "lasers": "488:1.0",
            "excitation_dichroic_probe_id": 1,
            "emission_splitters": [
                {"type": "Polarizer", "probe_id": None},
            ],
            "detectors": [
                {"name": "P", "qe_probe_id": 2},
                {"name": "S", "qe_probe_id": 3},
            ],
            "dyes": {"10": {"qy": 0.8, "ec": 90000}},
        }
        graph = build_easy_graph(config)
        splitter_nodes = [n for n in graph["nodes"] if n["type"] == "splitter"]
        polarizer_nodes = [n for n in splitter_nodes
                           if n["config"].get("splitter_type") == "Polarizer"]
        assert len(polarizer_nodes) == 1
        assert len([n for n in graph["nodes"] if n["type"] == "detector"]) == 2

    def test_graph_to_config_uses_detector_node_titles(self):
        """Graph presets should preserve visible detector titles in easy mode."""
        config = {
            "emission_splitters": [{"type": "Dichroic", "probe_id": 2}],
            "detectors": [
                {"name": "Stale Config 1", "bandpass_probe_id": 3, "qe_probe_id": 4},
                {"name": "Stale Config 2", "bandpass_probe_id": 5, "qe_probe_id": 6},
            ],
        }
        graph = build_easy_graph(config)
        detector_titles = ["Visible Green", "Visible Red"]
        for node, title in zip([n for n in graph["nodes"] if n["type"] == "detector"], detector_titles):
            node["title"] = title

        easy_config = _graph_to_config(graph)

        assert [det["name"] for det in easy_config["detectors"]] == detector_titles

    def test_normalize_repairs_missing_easy_topology_edges(self):
        """Loaded easy-mode graphs should regain missing splitter/filter edges."""
        config = {
            "emission_splitters": [{"type": "Dichroic", "probe_id": 2}],
            "detectors": [
                {"name": "Ch1", "bandpass_probe_id": 3, "qe_probe_id": 4},
                {"name": "Ch2", "bandpass_probe_id": 5, "qe_probe_id": 6},
            ],
        }
        graph = build_easy_graph(config)
        splitter = next(
            node for node in graph["nodes"]
            if node["type"] == "splitter" and node["title"].startswith("Dichroic Splitter")
        )
        filters = [node for node in graph["nodes"] if node["type"] == "filter"]
        graph["edges"] = [
            edge for edge in graph["edges"]
            if not (
                edge["source"] == splitter["id"]
                and edge["target"] in {node["id"] for node in filters}
            )
        ]

        normalized = normalize_lightpath_graph(graph)
        repaired_edges = {
            (edge["source"], edge["source_port"], edge["target"], edge["target_port"])
            for edge in normalized["edges"]
        }

        assert (splitter["id"], 1, filters[0]["id"], 0) in repaired_edges
        assert (splitter["id"], 2, filters[1]["id"], 0) in repaired_edges

    def test_normalize_converts_legacy_output_relative_ports(self):
        """Legacy output-relative source ports should become global port indices."""
        graph = build_easy_graph({
            "emission_splitters": [{"type": "Dichroic", "probe_id": 2}],
            "detectors": [
                {"name": "Ch1", "bandpass_probe_id": 3, "qe_probe_id": 4},
                {"name": "Ch2", "qe_probe_id": 5},
            ],
        })
        splitter = next(
            node for node in graph["nodes"]
            if node["type"] == "splitter" and node["title"].startswith("Dichroic Splitter")
        )
        edge = next(
            edge for edge in graph["edges"]
            if edge["source"] == splitter["id"] and edge["source_port"] == 1
        )
        edge["source_port"] = 0

        normalized = normalize_lightpath_graph(graph)
        normalized_edge = next(
            item for item in normalized["edges"]
            if item["source"] == splitter["id"] and item["target"] == edge["target"]
        )

        assert normalized_edge["source_port"] == 1

    def test_normalize_removes_legacy_forward_excitation_dichroic(self):
        """Older two-excitation-dichroic graphs should load as one dichroic."""
        graph = build_easy_graph({
            "emission_splitters": [{"type": "Dichroic", "probe_id": 2}],
            "detectors": [
                {"name": "Ch1", "qe_probe_id": 4},
                {"name": "Ch2", "qe_probe_id": 5},
            ],
        })
        sample = next(node for node in graph["nodes"] if node["type"] == "sample")
        light = next(node for node in graph["nodes"] if node["type"] == "light_source")
        exci = next(node for node in graph["nodes"] if node["title"] == "Excitation Dichroic")
        fw = {
            "id": "legacy-fw",
            "type": "splitter",
            "title": "Exci. Dichroic (FW)",
            "inputs": ["In"],
            "outputs": ["Transmission", "Reflection"],
            "config": {"probe_id": 1},
            "pos": [175.0, 50.0],
            "collapsed": False,
        }
        graph["nodes"].append(fw)
        graph["edges"] = [
            edge for edge in graph["edges"]
            if not (edge["source"] == light["id"] and edge["target"] == sample["id"])
        ]
        graph["edges"].extend([
            {"source": light["id"], "source_port": 0, "target": fw["id"], "target_port": 0},
            {"source": fw["id"], "source_port": 2, "target": sample["id"], "target_port": 0},
            {"source": sample["id"], "source_port": 1, "target": exci["id"], "target_port": 0},
        ])

        normalized = normalize_lightpath_graph(graph)

        assert all(node["title"] != "Exci. Dichroic (FW)" for node in normalized["nodes"])
        assert any(
            edge["source"] == light["id"] and edge["target"] == sample["id"]
            for edge in normalized["edges"]
        )

    def test_trailing_none_dichroic_does_not_create_extra_detector(self):
        """A non-required None dichroic row should not become a graph splitter."""
        graph = build_easy_graph({
            "emission_splitters": [{"type": "Dichroic", "probe_id": None}],
            "detectors": [{"name": "Only", "qe_probe_id": 4}],
        })

        emission_splitters = [
            node for node in graph["nodes"]
            if node["type"] == "splitter" and node["title"].startswith("Dichroic Splitter")
        ]
        detectors = [node for node in graph["nodes"] if node["type"] == "detector"]

        assert emission_splitters == []
        assert len(detectors) == 1

    def test_round_trip_json(self, tmp_path):
        """Config should survive a save/load round trip."""
        config = {
            "lasers": "488:1.0, 640:1.0",
            "excitation_dichroic_probe_id": 1,
            "emission_splitter_type": "Dichroic",
            "emission_splitter_probe_id": 2,
            "detectors": [
                {"name": "Ch1", "bandpass_probe_id": 3, "qe_probe_id": 4},
                {"name": "Ch2", "qe_probe_id": 5},
            ],
            "dyes": {"42": {"qy": 0.8, "ec": 90000}},
            "kappa2": 0.6667,
            "n": 1.33,
        }
        path = tmp_path / "preset.json"
        save_easy_preset(config, path)
        loaded = load_easy_preset(path)
        assert loaded["lasers"] == config["lasers"]
        assert loaded["excitation_dichroic_probe_id"] == config["excitation_dichroic_probe_id"]
        assert loaded["emission_splitter_type"] == config["emission_splitter_type"]
        assert loaded["emission_splitter_probe_id"] == config["emission_splitter_probe_id"]
        assert loaded["detectors"] == config["detectors"]
        assert loaded["dyes"] == config["dyes"]


class TestLastConfigPersistence:
    """Tests for auto-save / auto-load of the last config."""

    def test_save_and_load_last_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "chisurf.plugins.core.lightpath_simulator.gui.easy_mode.EASY_LAST_CONFIG_PATH",
            tmp_path / "lightpath_easy_last.json",
        )
        config = {"lasers": "561:1.0", "detectors": [], "dyes": {}}
        save_last_config(config)
        loaded = load_last_config()
        assert loaded == config

    def test_load_last_config_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "chisurf.plugins.core.lightpath_simulator.gui.easy_mode.EASY_LAST_CONFIG_PATH",
            tmp_path / "nonexistent.json",
        )
        assert load_last_config() is None


class TestTemplateLibrary:
    """Tests for built-in optical path template JSON files."""

    def test_template_library_is_one_file_per_optical_path(self):
        """The template directory should contain the requested template files."""
        files = {path.name for path in TEMPLATE_LIBRARY_DIR.glob("*.json")}

        assert files == {
            "2-color-2-detector.json",
            "3-color-3-detector.json",
            "anisotropy-2-detector.json",
            "polarizer-2xcolor-4-detector.json",
            "3-color-polarization-6-detector.json",
        }

    def test_template_detector_counts(self):
        """Each built-in template should build the advertised detector count."""
        expected_counts = {
            "two_color_2det": 2,
            "three_color_3det": 3,
            "anisotropy_2det": 2,
            "polarizer_2color_4det": 4,
            "three_color_polarization_6det": 6,
        }

        templates = load_template_library()

        assert {template["id"] for template in templates} == set(expected_counts)
        for template in templates:
            graph = build_easy_graph(template["config"])
            detectors = [node for node in graph["nodes"] if node["type"] == "detector"]
            assert len(detectors) == expected_counts[template["id"]]
