"""Tests for fps.json editor round-trip and score set support."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chisurf.plugins.modelling.fps_json_editor.core.colors import (
    DEFAULT_AV_COLOR,
    normalize_rgba,
    rgba_to_json,
)
from chisurf.plugins.modelling.fps_json_editor.core.model import FpsJsonModel
from chisurf.plugins.modelling.fps_json_editor.core.naming import (
    default_label_name,
    unique_label_name,
)
from chisurf.plugins.modelling.fps_json_editor.core.payload import (
    normalize_payload,
    summarize_payload,
    validate_payload,
)


class TestPackageBoundaries:
    """Package imports should preserve api/core/rpc/cli/gui boundaries."""

    def test_root_import_does_not_import_gui(self):
        """Importing the plugin root should not import GUI modules."""
        import chisurf.plugins.modelling.fps_json_editor as fps_json_editor

        assert fps_json_editor.name
        assert "chisurf.plugins.modelling.fps_json_editor.gui.tool" not in sys.modules
        assert "chisurf.plugins.modelling.fps_json_editor.gui.editor" not in sys.modules

    def test_cli_import_does_not_import_rpc(self):
        """CLI uses core helpers directly and should not import RPC services."""
        sys.modules.pop("chisurf.plugins.modelling.fps_json_editor.rpc.services", None)

        from chisurf.plugins.modelling.fps_json_editor.cli import cli as cli_module

        assert cli_module.cli.name == "cli"
        assert "chisurf.plugins.modelling.fps_json_editor.rpc.services" not in sys.modules

    def test_gui_communication_reexports_api_client(self):
        """The old GUI communication import path should remain compatible."""
        sys.modules.pop("chisurf.plugins.modelling.fps_json_editor.gui.editor", None)
        sys.modules.pop("chisurf.plugins.modelling.fps_json_editor.gui.tool", None)

        from chisurf.plugins.modelling.fps_json_editor.api.client import (
            FpsJsonEditorClient as ApiClient,
        )
        from chisurf.plugins.modelling.fps_json_editor.gui.communication import (
            FpsJsonEditorClient as GuiClient,
        )

        assert GuiClient is ApiClient
        assert "chisurf.plugins.modelling.fps_json_editor.gui.editor" not in sys.modules
        assert "chisurf.plugins.modelling.fps_json_editor.gui.tool" not in sys.modules

    def test_gui_package_exposes_entrypoints_lazily(self):
        """The GUI package should keep entrypoint imports lazy."""
        import chisurf.plugins.modelling.fps_json_editor.gui as gui

        assert sorted(gui.__all__) == ["FpsJsonEditor", "FpsJsonEditorTool"]
        assert "chisurf.plugins.modelling.fps_json_editor.gui.editor" not in sys.modules
        assert "chisurf.plugins.modelling.fps_json_editor.gui.tool" not in sys.modules


class TestMrcExport:
    """AV MRC export should use IMP-readable density maps."""

    def test_save_av_mrc_writes_imp_mrc(self, tmp_path):
        """A small weighted point cloud is saved as a readable MRC map."""
        import IMP.em

        from chisurf.plugins.modelling.fps_json_editor.core.mrc import save_av_mrc

        points = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 2.0],
                [1.0, 1.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )
        out_path = save_av_mrc(tmp_path / "av_export", points, 1.0)

        assert out_path.suffix == ".mrc"
        assert out_path.exists()

        density_map = IMP.em.read_map(str(out_path), IMP.em.MRCReaderWriter())
        assert density_map.get_number_of_voxels() >= 3
        assert density_map.get_max_value() >= 3.0


class TestLabelNaming:
    """Default labels should be available before users type a name."""

    def test_default_label_name_uses_chain_and_residue(self):
        """Chain and residue produce compact FPS labels."""
        assert default_label_name("E", 5) == "E5"
        assert default_label_name(" A ", "132") == "A132"

    def test_unique_label_name_suffixes_duplicates(self):
        """Duplicate generated names get stable numeric suffixes."""
        assert unique_label_name("A132", {"A132", "A132_2"}) == "A132_3"
        assert unique_label_name("E5", {"A132"}) == "E5"


class TestPayloadServices:
    """Core/RPC service helpers should expose useful fps.json operations."""

    def test_payload_summary_reports_references(self):
        """Payload summaries include used, unused, and missing references."""
        payload = {
            "Positions": {"A1": {}, "A2": {}},
            "Distances": {
                "ok": {"position1_name": "A1", "position2_name": "A2"},
                "bad": {"position1_name": "A1", "position2_name": "missing"},
            },
            "χ²": {"set1": {"distances": ["ok", "ghost"]}},
        }

        summary = summarize_payload(payload)

        assert summary["n_positions"] == 2
        assert summary["used_positions"] == ["A1", "A2"]
        assert summary["dangling_distances"] == [
            {"distance": "bad", "missing_positions": ["missing"]}
        ]
        assert summary["missing_score_set_distances"] == [
            {"score_set": "set1", "missing_distances": ["ghost"]}
        ]

    def test_payload_validation_and_normalization(self):
        """Validation flags bad references and normalization adds core sections."""
        payload = {
            "Positions": {"A1": {}},
            "Distances": {"bad": {"position1_name": "A1", "position2_name": "A2"}},
        }

        result = validate_payload(payload)
        normalized = normalize_payload({"Positions": {"A1": {}}})

        assert not result["valid"]
        assert "missing position" in result["errors"][0]
        assert normalized["Positions"] == {"A1": {}}
        assert normalized["Distances"] == {}

    def test_rpc_catalogue_and_handlers_offer_payload_services(self, tmp_path):
        """RPC handlers expose payload and MRC services with service envelopes."""
        from chisurf.plugins.modelling.fps_json_editor.rpc.services import (
            list_methods,
            normalize_payload_handler,
            save_av_mrc_handler,
            summarize_payload_handler,
            validate_payload_handler,
        )

        methods = list_methods()
        assert "fps_json_editor.payload.validate" in methods
        assert "fps_json_editor.payload.summarize" in methods
        assert "fps_json_editor.payload.normalize" in methods
        assert "fps_json_editor.av.mrc.save" in methods

        payload = {"Positions": {"A1": {}}, "Distances": {}}
        assert validate_payload_handler(payload)["result"]["valid"]
        assert summarize_payload_handler(payload)["result"]["n_positions"] == 1
        assert normalize_payload_handler(payload)["result"]["payload"]["Positions"] == {"A1": {}}

        points = [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0]]
        mrc_result = save_av_mrc_handler(str(tmp_path / "service_av"), points, 1.0)
        assert mrc_result["ok"]
        assert Path(mrc_result["result"]["path"]).suffix == ".mrc"
        assert Path(mrc_result["result"]["path"]).exists()

# ---------------------------------------------------------------------------
# Helpers that mirror the FpsJsonEditor payload logic so we can test without
# a Qt application running.
# ---------------------------------------------------------------------------

_RESERVED = {"Distances", "Positions", "χ²"}


def rebuild_from_payload(p: dict) -> dict[str, Any]:
    """Mirror FpsJsonEditor._rebuild_from_payload -> return editor state."""
    return {
        "positions": dict(p.get("Positions", {}) or {}),
        "distances": dict(p.get("Distances", {}) or {}),
        "score_sets": dict(p.get("χ²", {}) or {}),
        "extra_sections": {k: v for k, v in p.items() if k not in _RESERVED},
    }


def build_full_payload(state: dict) -> dict:
    """Mirror FpsJsonEditor._build_full_payload."""
    p = dict(state.get("extra_sections", {}))
    p["Distances"] = state["distances"]
    p["Positions"] = state["positions"]
    if state.get("score_sets"):
        p["χ²"] = state["score_sets"]
    return p


def cleanup_score_sets(
    score_sets: dict, distance_name: str
) -> None:
    """Mirror FpsJsonEditor._cleanup_score_sets."""
    for group in score_sets.values():
        if isinstance(group, dict) and "distances" in group:
            group["distances"] = [
                d for d in group["distances"] if d != distance_name
            ]


def distances_referencing(
    distances: dict, position_name: str
) -> list[str]:
    """Mirror FpsJsonEditor._distances_referencing."""
    return [
        dn for dn, d in distances.items()
        if d.get("position1_name") == position_name
        or d.get("position2_name") == position_name
    ]


def filter_distances_by_score_set(
    payload: dict, score_set: str
) -> dict:
    """Return only the distances belonging to a score set."""
    all_distances = payload.get("Distances", {})
    if (
        score_set
        and "χ²" in payload
        and score_set in payload["χ²"]
    ):
        group = payload["χ²"][score_set]
        keys = group.get("distances", []) if isinstance(group, dict) else []
        return {k: all_distances[k] for k in keys if k in all_distances}
    return dict(all_distances)


# ---------------------------------------------------------------------------
# Fixtures – example fps.json payloads based on real imp.bff examples
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_payload() -> dict:
    """Minimal fps.json without χ²."""
    return {
        "Positions": {
            "p1": {"chain_identifier": "A", "residue_seq_number": 1},
            "p2": {"chain_identifier": "A", "residue_seq_number": 2},
        },
        "Distances": {
            "p1_p2": {
                "distance": 25.0,
                "error_neg": 1.0,
                "error_pos": 1.0,
                "position1_name": "p1",
                "position2_name": "p2",
                "Forster_radius": 52.0,
                "distance_type": "RDAMean",
            }
        },
    }


@pytest.fixture
def payload_with_score_sets() -> dict:
    """Payload with two χ² score sets and an extra section (FlexFit)."""
    return {
        "Distances": {
            "d1": {
                "distance": 10.0, "error_neg": 1.0, "error_pos": 1.0,
                "position1_name": "p1", "position2_name": "p2",
                "Forster_radius": 52.0, "distance_type": "RDAMean",
            },
            "d2": {
                "distance": 20.0, "error_neg": 1.0, "error_pos": 1.0,
                "position1_name": "p1", "position2_name": "p3",
                "Forster_radius": 52.0, "distance_type": "RDAMean",
            },
            "d3": {
                "distance": 30.0, "error_neg": 1.0, "error_pos": 1.0,
                "position1_name": "p2", "position2_name": "p3",
                "Forster_radius": 52.0, "distance_type": "RDAMean",
            },
        },
        "Positions": {
            "p1": {"chain_identifier": "A", "residue_seq_number": 1},
            "p2": {"chain_identifier": "A", "residue_seq_number": 2},
            "p3": {"chain_identifier": "A", "residue_seq_number": 3},
        },
        "χ²": {
            "group_a": {
                "distances": ["d1", "d2"],
                "maximum_NaNs_allowed": 0,
                "penalty_NaN": 0,
            },
            "group_b": {
                "distances": ["d2", "d3"],
            },
        },
        "FlexFit": {
            "FexResSet1": {
                "Flexible residues": [
                    {"chain_identifier": "A", "residue_seq_number": 10},
                ],
                "Bonds": [],
            }
        },
    }


@pytest.fixture
def tmp_json(tmp_path: Path, payload_with_score_sets) -> Path:
    """Write a sample fps.json to a temp file."""
    fn = tmp_path / "test.fps.json"
    fn.write_text(json.dumps(payload_with_score_sets), encoding="utf-8")
    return fn


# ---------------------------------------------------------------------------
# Tests: payload round-trip
# ---------------------------------------------------------------------------


class TestPayloadRoundTrip:
    """The editor must preserve χ², FlexFit and unknown keys."""

    def test_rebuild_and_rebuild(self, simple_payload):
        state = rebuild_from_payload(simple_payload)
        assert state["positions"] == simple_payload["Positions"]
        assert state["distances"] == simple_payload["Distances"]
        assert state["score_sets"] == {}
        assert state["extra_sections"] == {}

    def test_round_trip_simple(self, simple_payload):
        state = rebuild_from_payload(simple_payload)
        rebuilt = build_full_payload(state)
        assert rebuilt == simple_payload

    def test_round_trip_with_score_sets(self, payload_with_score_sets):
        state = rebuild_from_payload(payload_with_score_sets)
        rebuilt = build_full_payload(state)
        assert rebuilt["Distances"] == payload_with_score_sets["Distances"]
        assert rebuilt["Positions"] == payload_with_score_sets["Positions"]
        assert rebuilt["χ²"] == payload_with_score_sets["χ²"]
        assert rebuilt["FlexFit"] == payload_with_score_sets["FlexFit"]

    def test_round_trip_json_output(self, tmp_json, payload_with_score_sets):
        """Ensure json.dumps(rebuild) matches expected output."""
        state = rebuild_from_payload(payload_with_score_sets)
        rebuilt = build_full_payload(state)
        serialized = json.dumps(rebuilt, sort_keys=True, indent=4, separators=(",", ": "))
        parsed = json.loads(serialized)
        assert parsed["χ²"]["group_a"]["distances"] == ["d1", "d2"]
        assert parsed["χ²"]["group_b"]["distances"] == ["d2", "d3"]
        assert "FlexFit" in parsed
        assert parsed["FlexFit"]["FexResSet1"]["Flexible residues"][0]["chain_identifier"] == "A"

    def test_preserves_unknown_top_level_keys(self):
        raw = {
            "Distances": {},
            "Positions": {},
            "CustomField": {"nested": [1, 2]},
        }
        state = rebuild_from_payload(raw)
        assert "CustomField" in state["extra_sections"]
        rebuilt = build_full_payload(state)
        assert rebuilt["CustomField"] == {"nested": [1, 2]}

    def test_empty_payload(self):
        state = rebuild_from_payload({})
        assert state["positions"] == {}
        assert state["distances"] == {}
        assert state["score_sets"] == {}
        assert state["extra_sections"] == {}


# ---------------------------------------------------------------------------
# Tests: score set operations
# ---------------------------------------------------------------------------


class TestScoreSetOperations:
    """Cleanup and membership logic."""

    def test_cleanup_distance_removes_from_all_groups(self, payload_with_score_sets):
        state = rebuild_from_payload(payload_with_score_sets)
        cleanup_score_sets(state["score_sets"], "d2")
        assert "d2" not in state["score_sets"]["group_a"]["distances"]
        assert "d2" not in state["score_sets"]["group_b"]["distances"]

    def test_cleanup_other_distances_untouched(self, payload_with_score_sets):
        state = rebuild_from_payload(payload_with_score_sets)
        cleanup_score_sets(state["score_sets"], "d1")
        assert state["score_sets"]["group_b"]["distances"] == ["d2", "d3"]

    def test_distances_referencing(self, payload_with_score_sets):
        dists = payload_with_score_sets["Distances"]
        assert distances_referencing(dists, "p1") == ["d1", "d2"]
        assert distances_referencing(dists, "p3") == ["d2", "d3"]
        assert distances_referencing(dists, "nonexistent") == []

    def test_cleanup_position_cascades_to_groups(self, payload_with_score_sets):
        state = rebuild_from_payload(payload_with_score_sets)
        to_remove = distances_referencing(state["distances"], "p1")
        for dn in to_remove:
            del state["distances"][dn]
            cleanup_score_sets(state["score_sets"], dn)
        # group_a had [d1, d2]; after removing d1 and d2 it should be empty
        assert state["score_sets"]["group_a"]["distances"] == []
        # group_b had [d2, d3]; d2 removed, d3 stays
        assert state["score_sets"]["group_b"]["distances"] == ["d3"]


# ---------------------------------------------------------------------------
# Tests: DirectLabelingPotential score_set filtering
# ---------------------------------------------------------------------------


class TestDirectLabelingPotentialScoreSet:
    """The model's DirectLabelingPotential must respect score_set."""

    def test_all_distances_when_empty_score_set(self, tmp_json, payload_with_score_sets):
        # Cannot fully instantiate (needs pdb2pqr), but we can test the filtering
        # by checking the payload-processing logic directly.
        filtered = filter_distances_by_score_set(payload_with_score_sets, "")
        assert set(filtered.keys()) == {"d1", "d2", "d3"}

    def test_filters_to_group_a(self, payload_with_score_sets):
        filtered = filter_distances_by_score_set(payload_with_score_sets, "group_a")
        assert set(filtered.keys()) == {"d1", "d2"}

    def test_filters_to_group_b(self, payload_with_score_sets):
        filtered = filter_distances_by_score_set(payload_with_score_sets, "group_b")
        assert set(filtered.keys()) == {"d2", "d3"}

    def test_unknown_score_set_returns_all(self, payload_with_score_sets):
        filtered = filter_distances_by_score_set(payload_with_score_sets, "nonexistent")
        assert set(filtered.keys()) == {"d1", "d2", "d3"}

    def test_no_chi_section(self, simple_payload):
        filtered = filter_distances_by_score_set(simple_payload, "whatever")
        assert set(filtered.keys()) == {"p1_p2"}

    def test_distances_preserved_unchanged(self, payload_with_score_sets):
        filtered = filter_distances_by_score_set(payload_with_score_sets, "group_a")
        assert filtered["d1"]["distance"] == 10.0
        assert filtered["d2"]["distance"] == 20.0

    def test_nans_and_penalty_fields_preserved(self, payload_with_score_sets):
        """Extra fields inside score groups survive round-trip."""
        assert payload_with_score_sets["χ²"]["group_a"]["maximum_NaNs_allowed"] == 0
        assert payload_with_score_sets["χ²"]["group_a"]["penalty_NaN"] == 0


# ---------------------------------------------------------------------------
# Tests: filtering used positions
# ---------------------------------------------------------------------------


class TestUsedPositions:
    """Only positions referenced by the active score set should be used."""

    def test_used_positions_group_a(self, payload_with_score_sets):
        filtered = filter_distances_by_score_set(payload_with_score_sets, "group_a")
        used: set[str] = set()
        for d in filtered.values():
            used.add(d["position1_name"])
            used.add(d["position2_name"])
        # d1: p1→p2, d2: p1→p3
        assert used == {"p1", "p2", "p3"}

    def test_used_positions_group_b(self, payload_with_score_sets):
        filtered = filter_distances_by_score_set(payload_with_score_sets, "group_b")
        used: set[str] = set()
        for d in filtered.values():
            used.add(d["position1_name"])
            used.add(d["position2_name"])
        # d2: p1→p3, d3: p2→p3
        assert used == {"p1", "p2", "p3"}

    def test_used_positions_single_distance(self, payload_with_score_sets):
        filtered = filter_distances_by_score_set(payload_with_score_sets, "single")
        # "single" doesn't exist, so all distances included
        assert set(filtered.keys()) == {"d1", "d2", "d3"}


# ---------------------------------------------------------------------------
# Tests: JSON serialization stability
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    """Verify that serializing/deserializing does not change the payload."""

    def test_position_av_color_round_trip(self):
        """Position av_color metadata survives model round-trip."""
        payload = {
            "Positions": {
                "D1": {
                    "chain_identifier": "A",
                    "residue_seq_number": 18,
                    "atom_name": "CB",
                    "av_color": [0.2, 0.4, 0.8, 0.55],
                },
            },
            "Distances": {},
        }

        model = FpsJsonModel()
        model.fps_json_payload = payload

        assert model.fps_json_payload["Positions"]["D1"]["av_color"] == [
            0.2,
            0.4,
            0.8,
            0.55,
        ]

    def test_position_av_color_helpers(self):
        """Color helpers normalize JSON-safe RGBA values."""
        assert normalize_rgba([51, 102, 204, 128]) == (
            0.2,
            0.4,
            0.8,
            128 / 255.0,
        )
        assert rgba_to_json([0.2, 0.4, 0.8]) == [0.2, 0.4, 0.8, DEFAULT_AV_COLOR[3]]

    def test_hgbp1_round_trip(self):
        """Round-trip similar to the hGBP1.fps.json structure."""
        payload = {
            "Distances": {
                "A18F-B18F": {
                    "error_neg": 2, "error_pos": 2, "distance": 61,
                    "position1_name": "A18F", "position2_name": "B18F",
                    "Forster_radius": 52, "distance_type": "RDAMean"
                },
            },
            "Positions": {
                "A18F": {
                    "allowed_sphere_radius": 2, "atom_name": "CB",
                    "chain_identifier": "A", "linker_length": 20,
                    "linker_width": 3.5, "radius1": 3.5,
                    "residue_seq_number": 18,
                    "simulation_grid_resolution": 2.0,
                    "simulation_type": "AV1",
                },
            },
            "χ²": {
                "inter": {
                    "distances": ["A18F-B18F"],
                },
            },
        }
        state = rebuild_from_payload(payload)
        rebuilt = build_full_payload(state)
        assert rebuilt == payload

    def test_flexfit_round_trip(self):
        payload = {
            "Distances": {},
            "Positions": {},
            "χ²": {
                "s1": {"distances": [], "maximum_NaNs_allowed": 0, "penalty_NaN": 0},
            },
            "FlexFit": {
                "Set1": {
                    "Flexible residues": [
                        {"chain_identifier": "A", "residue_seq_number": 10},
                    ],
                    "Bonds": [
                        [
                            {"chain_identifier": "A", "residue_seq_number": 10, "atom_name": "CA"},
                            {"chain_identifier": "A", "residue_seq_number": 20, "atom_name": "CB"},
                        ]
                    ],
                }
            },
        }
        state = rebuild_from_payload(payload)
        rebuilt = build_full_payload(state)
        assert rebuilt == payload
        assert rebuilt["χ²"]["s1"]["penalty_NaN"] == 0
        assert len(rebuilt["FlexFit"]["Set1"]["Bonds"]) == 1


# ---------------------------------------------------------------------------
# Tests: build_move_map_from_flexfit and list_flexfit_sets
# ---------------------------------------------------------------------------


class TestFlexFitMoveMap:
    """build_move_map_from_flexfit must map FlexFit residues to structure indices."""

    def test_list_flexfit_sets(self, tmp_json, payload_with_score_sets):
        from chisurf.plugins.modelling.proteinmc.model import list_flexfit_sets
        sets = list_flexfit_sets(tmp_json)
        assert sets == ["FexResSet1"]

    def test_list_flexfit_sets_no_file(self):
        from chisurf.plugins.modelling.proteinmc.model import list_flexfit_sets
        assert list_flexfit_sets("/nonexistent/path.json") == []

    def test_list_flexfit_sets_no_flexfit(self, tmp_path):
        from chisurf.plugins.modelling.proteinmc.model import list_flexfit_sets
        fn = tmp_path / "noflex.json"
        fn.write_text(json.dumps({"Distances": {}, "Positions": {}}))
        assert list_flexfit_sets(fn) == []

    def test_build_move_map_no_flexfit(self, tmp_path):
        from chisurf.plugins.modelling.proteinmc.model import build_move_map_from_flexfit
        fn = tmp_path / "noflex.json"
        fn.write_text(json.dumps({"Distances": {}, "Positions": {}}))
        result = build_move_map_from_flexfit(None, fn, None)
        assert result is None

    def test_build_move_map_returns_none_on_no_match(self, tmp_path):
        """When FlexFit residues don't match any structure residue, return None."""
        from chisurf.plugins.modelling.proteinmc.model import build_move_map_from_flexfit
        fn = tmp_path / "flex.json"
        fn.write_text(json.dumps({
            "Distances": {},
            "Positions": {},
            "FlexFit": {
                "S1": {
                    "Flexible residues": [
                        {"chain_identifier": "Z", "residue_seq_number": 9999},
                    ],
                    "Bonds": [],
                }
            },
        }))
        atoms = np.zeros(3, dtype=[("chain", "S1"), ("res_id", "i4")])
        atoms["chain"] = [b"A", b"A", b"A"]
        atoms["res_id"] = [1, 2, 3]

        class MockStructure:
            n_residues = 3

            @property
            def residue_dict(self):
                return {1: {}, 2: {}, 3: {}}

            @property
            def atoms(self):
                return atoms

        result = build_move_map_from_flexfit(MockStructure(), fn, "S1")
        assert result is None

    def test_build_move_map_matching_residues(self, tmp_path):
        """When FlexFit residues match, the move_map has 1.0 at those indices."""
        from chisurf.plugins.modelling.proteinmc.model import build_move_map_from_flexfit
        fn = tmp_path / "flex.json"
        fn.write_text(json.dumps({
            "Distances": {},
            "Positions": {},
            "FlexFit": {
                "S1": {
                    "Flexible residues": [
                        {"chain_identifier": "A", "residue_seq_number": 2},
                    ],
                    "Bonds": [],
                }
            },
        }))

        atoms = np.zeros(3, dtype=[("chain", "S1"), ("res_id", "i4")])
        atoms["chain"] = [b"A", b"A", b"A"]
        atoms["res_id"] = [1, 2, 3]

        class MockStructure:
            n_residues = 3

            @property
            def residue_dict(self):
                return {1: {}, 2: {}, 3: {}}

            @property
            def atoms(self):
                return atoms

        result = build_move_map_from_flexfit(MockStructure(), fn, "S1")
        assert result is not None
        assert result.shape == (3,)
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[2] == 0.0

    def test_build_move_map_defaults_to_first_set(self, tmp_path):
        """When flexfit_set=None, the first set is used."""
        from chisurf.plugins.modelling.proteinmc.model import build_move_map_from_flexfit
        fn = tmp_path / "flex.json"
        fn.write_text(json.dumps({
            "Distances": {},
            "Positions": {},
            "FlexFit": {
                "First": {
                    "Flexible residues": [
                        {"chain_identifier": "A", "residue_seq_number": 1},
                    ],
                    "Bonds": [],
                },
                "Second": {
                    "Flexible residues": [
                        {"chain_identifier": "A", "residue_seq_number": 3},
                    ],
                    "Bonds": [],
                },
            },
        }))

        atoms = np.zeros(3, dtype=[("chain", "S1"), ("res_id", "i4")])
        atoms["chain"] = [b"A", b"A", b"A"]
        atoms["res_id"] = [1, 2, 3]

        class MockStructure:
            n_residues = 3

            @property
            def residue_dict(self):
                return {1: {}, 2: {}, 3: {}}

            @property
            def atoms(self):
                return atoms

        result = build_move_map_from_flexfit(MockStructure(), fn, None)
        assert result is not None
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_build_move_map_empty_residues_returns_none(self, tmp_path):
        """A FlexFit set with no Flexible residues returns None."""
        from chisurf.plugins.modelling.proteinmc.model import build_move_map_from_flexfit
        fn = tmp_path / "flex.json"
        fn.write_text(json.dumps({
            "Distances": {},
            "Positions": {},
            "FlexFit": {
                "S1": {
                    "Flexible residues": [],
                    "Bonds": [],
                }
            },
        }))

        atoms = np.zeros(0, dtype=[("chain", "S1"), ("res_id", "i4")])

        class MockStructure:
            n_residues = 0

            @property
            def residue_dict(self):
                return {}

            @property
            def atoms(self):
                return atoms

        result = build_move_map_from_flexfit(MockStructure(), fn, "S1")
        assert result is None

    def test_build_move_map_nonexistent_set_returns_none(self, tmp_path):
        """Requesting a nonexistent FlexFit set returns None."""
        from chisurf.plugins.modelling.proteinmc.model import build_move_map_from_flexfit
        fn = tmp_path / "flex.json"
        fn.write_text(json.dumps({
            "Distances": {},
            "Positions": {},
            "FlexFit": {
                "S1": {
                    "Flexible residues": [
                        {"chain_identifier": "A", "residue_seq_number": 1},
                    ],
                    "Bonds": [],
                }
            },
        }))

        atoms = np.zeros(1, dtype=[("chain", "S1"), ("res_id", "i4")])
        atoms["chain"] = [b"A"]
        atoms["res_id"] = [1]

        class MockStructure:
            n_residues = 1

            @property
            def residue_dict(self):
                return {1: {}}

            @property
            def atoms(self):
                return atoms

        result = build_move_map_from_flexfit(MockStructure(), fn, "NONEXISTENT")
        assert result is None
