"""Easy mode for the Light Path Simulator — form-based optical configuration."""

import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.settings import cs_settings
from chisurf.plugins.core.lightpath_simulator.core.workflow import (
    MFDatabaseAdapter,
    _simulate_with_db,
    resolve_db_path,
)

from chisurf.gui.widgets.filtered_table import FilteredTableWidget

logger = logging.getLogger(__name__)

EASY_LAST_CONFIG_PATH = Path.home() / ".chisurf" / "settings" / "lightpath_easy_last.json"
EASY_PRESETS_DIR = Path.home() / ".chisurf" / "presets" / "lightpath_easy"
OPTICAL_PRESETS_DIR = Path.home() / ".chisurf" / "presets" / "lightpath_optical"
DYE_PRESETS_DIR = Path.home() / ".chisurf" / "presets" / "lightpath_dyes"
TEMPLATE_LIBRARY_DIR = Path(__file__).resolve().parents[1] / "templates"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_pid(pid):
    if pid is None:
        return None
    try:
        return int(pid)
    except (ValueError, TypeError):
        return pid


def _make_combo(parent, probes, filter_key):
    combo = QtWidgets.QComboBox(parent)
    combo.setEditable(True)
    combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
    if combo.completer():
        combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        combo.completer().setFilterMode(QtCore.Qt.MatchContains)
    combo.addItem("None", None)
    for p in probes:
        match = False
        if not filter_key:
            match = True
        elif filter_key.startswith("category:"):
            cats = filter_key.split(":", 1)[1].split(",")
            match = p.get("category") in cats
        else:
            match = bool(p.get(filter_key))
            
        if match:
            combo.addItem(p["name"], p["probe_id"])
    return combo


def _combo_value(combo):
    return combo.itemData(combo.currentIndex()) if combo else None


def _set_combo(combo, pid):
    if pid is not None and combo is not None:
        idx = combo.findData(_normalize_pid(pid))
        if idx >= 0:
            combo.setCurrentIndex(idx)


def load_template_library(directory: str | Path = TEMPLATE_LIBRARY_DIR) -> list[dict]:
    """Load built-in optical path templates from one JSON file per template."""
    template_dir = Path(directory)
    if not template_dir.exists():
        return []

    valid_templates = []
    for path in sorted(template_dir.glob("*.json")):
        try:
            with open(path) as handle:
                template = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load lightpath template %s: %s", path, exc)
            continue
        if not isinstance(template, dict):
            continue
        config = template.get("config")
        if not isinstance(config, dict):
            continue
        template.setdefault("id", path.stem)
        template.setdefault("name", path.stem)
        template["_path"] = str(path)
        valid_templates.append(template)
    return valid_templates


def _node_display_name(node: dict, config_key: str, default: str) -> str:
    """Return the visible node title before falling back to config metadata."""
    title = str(node.get("title") or "").strip()
    if title:
        return title
    return str(node.get("config", {}).get(config_key) or default)


def _clean_splitters_for_detector_count(
    splitters: list[dict],
    detector_count: int,
) -> list[dict]:
    """Drop optional trailing placeholder dichroics that do not affect topology."""
    cleaned = [dict(splitter) for splitter in splitters]
    while cleaned and len(cleaned) >= max(detector_count, 1):
        last = cleaned[-1]
        splitter_type = str(last.get("type") or "Dichroic")
        probe_id = _normalize_pid(last.get("probe_id"))
        if splitter_type != "Dichroic" or probe_id is not None:
            break
        cleaned.pop()
    return cleaned


def _abbreviate_detector_name(name: str, color_index: int, pol_index: int) -> str:
    """Abbreviate detector names to compact form like ``C1_VV``, ``C2_VH``.

    Parameters
    ----------
    name : str
        Original detector name (e.g. ``"Color 1 Parallel"``).
    color_index : int
        1-based color channel index.
    pol_index : int
        Polarizer index (0 = parallel/VV, 1 = perpendicular/VH).

    Returns
    -------
    str
        Abbreviated name such as ``"C1_VV"`` or ``"C2_VH"``.
    """
    low = name.lower()
    if "parallel" in low or "perpendicular" in low:
        suffix = "_VV" if "parallel" in low else "_VH"
        return f"C{color_index}{suffix}"
    return f"C{color_index}"


def _is_polarizer_template(detectors: list[dict]) -> bool:
    """Return True if any detector name contains Parallel/Perpendicular."""
    for d in detectors:
        low = str(d.get("name", "")).lower()
        if "parallel" in low or "perpendicular" in low:
            return True
    return False


def _split_detector_name(name: str) -> tuple[int, str]:
    """Parse ``'Color N Parallel'`` → ``(N, 'VV')`` or return ``(0, name)``."""
    low = name.lower()
    color_idx = 0
    if "color" in low:
        parts = low.split()
        for part in parts:
            if part.isdigit():
                color_idx = int(part)
                break
    if "parallel" in low:
        pol = "VV"
    elif "perpendicular" in low:
        pol = "VH"
    else:
        pol = ""
    return color_idx, pol


def _make_combo(parent, probes, filter_key):
    combo = QtWidgets.QComboBox(parent)
    combo.setEditable(True)
    combo.addItem("None", None)
    for p in probes:
        if p.get(filter_key):
            combo.addItem(p["name"], p["probe_id"])
    return combo


class _SingleProbeTable(QtWidgets.QWidget):
    """Compact single-select probe table with spectra tooltips using filterable table."""

    changed = QtCore.Signal()

    def __init__(self, probes, db_path=None, filter_key=None, parent=None, show_header=True, header_text="Probe"):
        super().__init__(parent)
        self.probes = probes
        self._db_path = db_path
        self._filter_key = filter_key
        self._header_text = header_text
        self._setup_ui()
        self.populate()  # Populate table on creation

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.filtered_table = FilteredTableWidget(parent=self)
        self.filtered_table.set_item_factory(lambda item, key: _SpectraTooltipItem(
            name=item.get("name", ""),
            key=key,
            render_fn=self._tooltip_html
        ))
        self.filtered_table.selectionChanged.connect(self.changed.emit)
        layout.addWidget(self.filtered_table)

    def _tooltip_html(self, key: int) -> str:
        img = _render_spectra_thumbnail(key, self._ensure_db_adapter())
        if not img:
            return ""
        name = ""
        for p in self.probes:
            if p.get("probe_id") == key:
                name = p.get("name", "")
                break
        return f"<div style='text-align:center'><b>{name}</b><br>{img}</div>"

    def _ensure_db_adapter(self):
        if not hasattr(self, "_db_adapter"):
            self._db_adapter = None
            self._db_opened = False
        if not self._db_opened and self._db_path:
            self._db_opened = True
            try:
                self._db_adapter = MFDatabaseAdapter(MFDatabase(self._db_path))
            except Exception:
                self._db_adapter = None
        return self._db_adapter

    def populate(self, selected_probe_id=None):
        # Filter probes by filter_key if specified
        if self._filter_key:
            if self._filter_key.startswith("category:"):
                cats = self._filter_key.split(":", 1)[1].split(",")
                items = [p for p in self.probes if p.get("category") in cats]
            else:
                items = [p for p in self.probes if p.get(self._filter_key)]
        else:
            items = self.probes
        self.filtered_table.set_data(items, key_fn=lambda x: x.get("probe_id"))
        self.filtered_table.set_filter_fn(lambda item, text: text.lower() in item.get("name", "").lower())
        self.filtered_table.set_selected(selected_probe_id)

    def get_selected_probe_id(self):
        return self.filtered_table.get_selected()

    def set_selected_probe_id(self, pid):
        self.filtered_table.set_selected(pid)

    def close_db(self):
        if hasattr(self, "_db_adapter") and self._db_adapter is not None:
            try:
                self._db_adapter.db.close()
            except Exception:
                pass
            self._db_adapter = None

    def set_header_text(self, text: str) -> None:
        """Change the table column header text (acts as the row label)."""
        self.filtered_table.table.setHorizontalHeaderLabels([text])


# ---------------------------------------------------------------------------
# Graph ↔ config conversion
# ---------------------------------------------------------------------------

def _graph_to_config(graph: dict) -> dict:
    """Convert a GraphDef graph dict back into an easy-mode config dict.

    Handles presets saved from the Full Simulator (which store the full
    node/edge graph) so the Easy Mode form can populate correctly.
    """
    nodes_list = graph.get("nodes", [])
    edges_list = graph.get("edges", [])
    nodes_map: dict[str, dict] = {n["id"]: n for n in nodes_list}

    # Build forward adjacency: source_id → [(source_port, target_id, target_port)]
    forward: dict[str, list[tuple[int, str, int]]] = {}
    for e in edges_list:
        src = e["source"]
        forward.setdefault(src, []).append(
            (e["source_port"], e["target"], e["target_port"])
        )

    def node_type_of(nid: str) -> str:
        nd = nodes_map.get(nid)
        return nd["type"] if nd else ""

    config: dict = {}

    # --- Lasers ---
    for n in nodes_list:
        if n["type"] == "light_source":
            config["lasers"] = n.get("config", {}).get("manual_lines", "488:1.0, 640:1.0")
            break

    # --- Dyes ---
    for n in nodes_list:
        if n["type"] == "sample":
            dye_props = n.get("config", {}).get("dye_properties", {})
            if dye_props:
                config["dyes"] = dye_props
            else:
                pids = n.get("config", {}).get("probe_ids", [])
                if pids:
                    config["dyes"] = {str(pid): {"qy": 1.0, "ec": 1.0} for pid in pids}
            break

    # --- Förster parameters ---
    for n in nodes_list:
        if n["type"] == "forster_radius":
            fcfg = n.get("config", {})
            config["kappa2"] = fcfg.get("kappa2", 0.6667)
            config["n"] = fcfg.get("n", 1.33)
            break

    # --- Classify splitters ---
    sample_id: str | None = None
    for n in nodes_list:
        if n["type"] == "sample":
            sample_id = n["id"]
            break

    # Splitters fed by Sample's output port 1 are excitation dichroics
    exci_splitter_ids: set[str] = set()
    exci_bw_id: str | None = None
    if sample_id is not None:
        for sp, tgt, tp in forward.get(sample_id, []):
            if sp == 1 and node_type_of(tgt) == "splitter":
                exci_splitter_ids.add(tgt)
                exci_bw_id = tgt
                break

    # Excitation dichroic probe (from ExciBW)
    if exci_bw_id is not None:
        pid = nodes_map[exci_bw_id].get("config", {}).get("probe_id")
        if pid is not None:
            config["excitation_dichroic_probe_id"] = pid

    # --- Walk emission cascade from ExciBW's transmission (port 1) ---
    splitters: list[dict] = []
    detectors: list[dict] = []

    def _add_detector_from_chain(start_id: str) -> None:
        """Resolve a filter→detector chain and add the detector."""
        dname, bp_pid, qe_pid = _resolve_detector_chain(
            start_id, nodes_map, forward
        )
        detectors.append({
            "name": dname,
            "bandpass_probe_id": bp_pid,
            "qe_probe_id": qe_pid,
        })

    def _walk_splitter(splitter_id: str, visited: set[str]) -> str | None:
        """Record one emission splitter and find the next one."""
        nd = nodes_map.get(splitter_id, {})
        cfg = nd.get("config", {})
        splitters.append({
            "type": cfg.get("splitter_type", "Dichroic"),
            "probe_id": cfg.get("probe_id"),
        })
        # Detector on transmission port (port 1)
        for sp, tgt, tp in forward.get(splitter_id, []):
            if sp == 1:
                _add_detector_from_chain(tgt)
                break
        # Next splitter on reflection port (port 2)
        for sp, tgt, tp in forward.get(splitter_id, []):
            if sp == 2:
                next_type = node_type_of(tgt)
                if next_type == "splitter" and tgt not in visited:
                    return tgt
                # Reflection feeds a detector chain directly (last splitter)
                if next_type in ("filter", "detector"):
                    _add_detector_from_chain(tgt)
                break
        return None

    if exci_bw_id is not None:
        # Walk from ExciBW's transmission port
        for sp, tgt, tp in forward.get(exci_bw_id, []):
            if sp == 1:
                cur = tgt
                visited: set[str] = set()
                while cur is not None and cur not in visited:
                    visited.add(cur)
                    nt = node_type_of(cur)
                    if nt == "splitter":
                        nxt = _walk_splitter(cur, visited)
                        cur = nxt
                    elif nt in ("filter", "detector"):
                        # Direct connection to detector chain (no splitters)
                        _add_detector_from_chain(cur)
                        break
                    else:
                        # Unknown — follow single outgoing edge
                        nxt = None
                        for sp2, tgt2, tp2 in forward.get(cur, []):
                            nxt = tgt2
                            break
                        cur = nxt
                break

    # --- Fallbacks when edges are missing or cascade walk found nothing ---

    # Fallback 1: collect ALL detectors from the node list
    found_det_names = {d["name"] for d in detectors}
    for n in nodes_list:
        if n["type"] == "detector":
            dname = _node_display_name(n, "detector_name", "Detector")
            if dname not in found_det_names:
                detectors.append({
                    "name": dname,
                    "bandpass_probe_id": None,
                    "qe_probe_id": n.get("config", {}).get("probe_id"),
                })
                found_det_names.add(dname)

    # Fallback 2: if no splitters found but detectors exist, infer the
    # splitter that feeds them (the excitation dichroic itself is the
    # emission splitter in single-splitter topologies)
    if not splitters and len(detectors) >= 2:
        exci_pid = config.get("excitation_dichroic_probe_id")
        splitters.append({"type": "Dichroic", "probe_id": exci_pid})

    config["emission_splitters"] = splitters
    config["detectors"] = detectors
    return config


def _resolve_detector_chain(
    start_id: str,
    nodes_map: dict[str, dict],
    forward: dict[str, list[tuple[int, str, int]]],
) -> tuple[str, Any, Any]:
    """Walk from a node to find the detector, its bandpass, and QE probe."""
    cur = start_id
    bp_pid = None
    qe_pid = None
    det_name = "Channel"

    visited: set[str] = set()
    while cur and cur not in visited:
        visited.add(cur)
        nd = nodes_map.get(cur)
        if nd is None:
            break
        ntype = nd["type"]
        cfg = nd.get("config", {})

        if ntype == "filter":
            bp_pid = cfg.get("probe_id")

        elif ntype == "detector":
            det_name = _node_display_name(nd, "detector_name", "Detector")
            qe_pid = cfg.get("probe_id")
            break

        # Follow the single outgoing edge
        next_id: str | None = None
        for sp, tgt, tp in forward.get(cur, []):
            next_id = tgt
            break
        cur = next_id

    return det_name, bp_pid, qe_pid


def normalize_lightpath_graph(graph: dict) -> dict:
    """Return a graph dict with legacy port indices and easy edges repaired."""
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        return graph

    normalized = copy.deepcopy(graph)
    _normalize_legacy_edge_ports(normalized)
    _repair_easy_topology_edges(normalized)
    return normalized


def _normalize_legacy_edge_ports(graph: dict) -> None:
    """Convert output-relative source ports to NodeScene global port indices."""
    nodes = {node.get("id"): node for node in graph.get("nodes", [])}
    for edge in graph.get("edges", []):
        src = nodes.get(edge.get("source"))
        dst = nodes.get(edge.get("target"))
        if src is None or dst is None:
            continue
        try:
            source_port = int(edge.get("source_port", 0))
            target_port = int(edge.get("target_port", 0))
        except (TypeError, ValueError):
            continue

        src_inputs = len(src.get("inputs", []))
        src_outputs = len(src.get("outputs", []))
        dst_inputs = len(dst.get("inputs", []))
        source_is_input = source_port < src_inputs
        target_is_input = target_port < dst_inputs
        source_could_be_output_relative = 0 <= source_port < src_outputs

        if source_is_input and target_is_input and source_could_be_output_relative:
            edge["source_port"] = src_inputs + source_port


def _repair_easy_topology_edges(graph: dict) -> None:
    """Restore missing edges in graphs generated by the easy-mode topology."""
    nodes = graph.get("nodes", [])
    edges = graph.setdefault("edges", [])
    if not isinstance(edges, list):
        graph["edges"] = []
        edges = graph["edges"]

    by_type: dict[str, list[dict]] = {}
    for node in nodes:
        by_type.setdefault(str(node.get("type", "")), []).append(node)

    splitters = by_type.get("splitter", [])
    exci_fw = _find_titled_node(splitters, "exci", "fw")
    exci = _find_titled_node(splitters, "exci", "bw")
    if exci is None:
        exci = _find_titled_node(splitters, "excitation", "dichroic")
    if exci is None:
        return

    light = _first_node(by_type, "light_source")
    sample = _first_node(by_type, "sample")
    forster = _first_node(by_type, "forster_radius")
    detectors = _sort_channel_nodes(by_type.get("detector", []))
    if light is None or sample is None or not detectors:
        return

    if exci_fw is not None and exci_fw in nodes:
        nodes.remove(exci_fw)
        edges[:] = [
            edge for edge in edges
            if edge.get("source") != exci_fw.get("id") and edge.get("target") != exci_fw.get("id")
        ]

    emission_splitters = _sort_channel_nodes(
        [node for node in splitters if node not in (exci, exci_fw)]
    )
    filter_by_detector = _match_filters_to_detectors(
        by_type.get("filter", []),
        detectors,
    )

    def add_edge(source: dict, source_port: int, target: dict, target_port: int) -> None:
        """Add an edge if the referenced ports exist and no duplicate exists."""
        if not _has_port(source, source_port) or not _has_port(target, target_port):
            return
        key = (source.get("id"), source_port, target.get("id"), target_port)
        for existing in list(edges):
            existing_key = (
                existing.get("source"),
                existing.get("source_port"),
                existing.get("target"),
                existing.get("target_port"),
            )
            if existing_key == key:
                return
            if (
                existing.get("source") == source.get("id")
                and existing.get("target") == target.get("id")
                and existing.get("target_port") == target_port
            ):
                edges.remove(existing)
        edges.append({
            "source": source.get("id"),
            "source_port": source_port,
            "target": target.get("id"),
            "target_port": target_port,
        })

    add_edge(light, 0, sample, 0)
    add_edge(sample, 1, exci, 0)
    if forster is not None:
        add_edge(sample, 2, forster, 0)

    previous = exci
    previous_port = 1
    for index, splitter in enumerate(emission_splitters):
        add_edge(previous, previous_port, splitter, 0)
        _add_detector_chain_edge(add_edge, splitter, 1, detectors[index], filter_by_detector)
        previous = splitter
        previous_port = 2

    last_index = len(emission_splitters)
    if last_index < len(detectors):
        _add_detector_chain_edge(
            add_edge,
            previous,
            previous_port,
            detectors[last_index],
            filter_by_detector,
        )


def _find_titled_node(nodes: list[dict], *needles: str) -> dict | None:
    """Find the first node whose title contains every needle."""
    for node in nodes:
        title = str(node.get("title") or "").lower()
        if all(needle.lower() in title for needle in needles):
            return node
    return None


def _first_node(nodes_by_type: dict[str, list[dict]], node_type: str) -> dict | None:
    """Return the first node of a type if one exists."""
    nodes = nodes_by_type.get(node_type) or []
    return nodes[0] if nodes else None


def _sort_channel_nodes(nodes: list[dict]) -> list[dict]:
    """Sort channel-like nodes by embedded number and then by screen position."""
    def key(node: dict) -> tuple[int, float, float, str]:
        title = str(node.get("title") or node.get("config", {}).get("detector_name") or "")
        number = _extract_first_int(title)
        pos = node.get("pos") or [0.0, 0.0]
        try:
            x_pos = float(pos[0])
            y_pos = float(pos[1])
        except (TypeError, ValueError, IndexError):
            x_pos = 0.0
            y_pos = 0.0
        return (number if number is not None else 10_000, y_pos, x_pos, title)

    return sorted(nodes, key=key)


def _extract_first_int(text: str) -> int | None:
    """Return the first integer embedded in text, if present."""
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else None


def _match_filters_to_detectors(
    filters: list[dict],
    detectors: list[dict],
) -> dict[str, dict]:
    """Match bandpass filters to detectors by title, falling back to row order."""
    result: dict[str, dict] = {}
    remaining = list(filters)
    for detector in detectors:
        detector_name = _node_display_name(detector, "detector_name", "Detector")
        match = None
        for candidate in remaining:
            title = str(candidate.get("title") or "")
            if detector_name and detector_name.lower() in title.lower():
                match = candidate
                break
        if match is not None:
            result[str(detector.get("id"))] = match
            remaining.remove(match)

    if remaining:
        for detector, candidate in zip(detectors, _sort_channel_nodes(remaining)):
            result.setdefault(str(detector.get("id")), candidate)
    return result


def _has_port(node: dict, port_index: int) -> bool:
    """Return whether a global NodeScene port index exists on a node."""
    port_count = len(node.get("inputs", [])) + len(node.get("outputs", []))
    return 0 <= port_index < port_count


def _add_detector_chain_edge(
    add_edge,
    source: dict,
    source_port: int,
    detector: dict,
    filter_by_detector: dict[str, dict],
) -> None:
    """Add source-to-filter-to-detector edges for one detector channel."""
    bandpass = filter_by_detector.get(str(detector.get("id")))
    if bandpass is None:
        add_edge(source, source_port, detector, 0)
        return
    add_edge(source, source_port, bandpass, 0)
    add_edge(bandpass, 1, detector, 0)


# ---------------------------------------------------------------------------
# Graph builder — constructs a GraphDef-compatible dict
# ---------------------------------------------------------------------------

def _node(
    id: str, type_: str, title: str, inputs: list, outputs: list,
    config: dict, pos: tuple[float, float],
) -> dict:
    """Build a validated node dict (auto-adds collapsed and version)."""
    return {
        "id": id,
        "type": type_,
        "title": title,
        "inputs": inputs,
        "outputs": outputs,
        "config": config,
        "pos": list(pos),
        "collapsed": False,
    }


def build_easy_graph(config: dict) -> dict:
    """Build a GraphDef-compatible dict from an easy-mode preset config.

    Optical path:
      Light Source → Sample
      Sample → Excitation Dichroic → cascaded Emission Splitters → N Detectors
      Sample → Förster Radius

    Supports two config formats:
    - New: ``emission_splitters`` (list of ``{type, probe_id}``)
    - Legacy: ``emission_splitter_probe_id`` + ``emission_splitter_type`` (single splitter)
    """
    nodes = []
    edges = []

    light_id = str(uuid.uuid4())
    sample_id = str(uuid.uuid4())
    exci_id = str(uuid.uuid4())
    forster_id = str(uuid.uuid4())

    # 1. Light source
    lasers = config.get("lasers", "488:1.0, 640:1.0")
    nodes.append(_node(
        id=light_id, type_="light_source", title="Light Source",
        inputs=[], outputs=["Light"],
        config={"source_mode": "manual", "manual_lines": lasers},
        pos=(50.0, 200.0),
    ))

    # 2. Sample
    dye_ids = []
    dye_props = {}
    for pid_str, props in config.get("dyes", {}).items():
        pid = _normalize_pid(pid_str)
        if pid is not None:
            dye_ids.append(pid)
            dye_props[pid_str] = props
    nodes.append(_node(
        id=sample_id, type_="sample", title="Sample / Fluorophore",
        inputs=["In"], outputs=["Out", "Dye Data"],
        config={
            "probe_ids": dye_ids,
            "probe_id": dye_ids[0] if dye_ids else None,
            "dye_properties": dye_props,
        },
        pos=(300.0, 200.0),
    ))

    # 3. Excitation dichroic — emission path
    exci_pid = _normalize_pid(config.get("excitation_dichroic_probe_id"))
    nodes.append(_node(
        id=exci_id, type_="splitter",
        title="Excitation Dichroic",
        inputs=["In"], outputs=["Transmission", "Reflection"],
        config={"probe_id": exci_pid},
        pos=(500.0, 200.0),
    ))
    edges.append({"source": light_id, "source_port": 0, "target": sample_id, "target_port": 0})
    edges.append({"source": sample_id, "source_port": 1, "target": exci_id, "target_port": 0})

    # 4. Emission splitters (cascaded) → Detector channels
    splitters = config.get("emission_splitters", [])
    if not splitters:
        # Legacy: single splitter from emission_splitter_probe_id + emission_splitter_type
        legacy_pid = _normalize_pid(config.get("emission_splitter_probe_id"))
        legacy_type = config.get("emission_splitter_type", "Dichroic")
        if legacy_pid is not None:
            splitters = [{"type": legacy_type, "probe_id": legacy_pid}]

    detectors = config.get("detectors", [])
    splitters = _clean_splitters_for_detector_count(splitters, len(detectors))
    n_detectors = len(splitters) + 1  # N splitters → N+1 detectors

    # Auto-generate splitters if more detectors than splitters allow
    while n_detectors < len(detectors):
        splitters.append({"type": "Dichroic", "probe_id": None})
        n_detectors = len(splitters) + 1

    # Ensure detector list has enough entries
    while len(detectors) < n_detectors:
        detectors.append({"name": f"Channel {len(detectors) + 1}"})

    # Build cascaded splitters
    prev_node_id = exci_id
    prev_port = 1  # Excitation dichroic transmission output
    splitter_ids = []

    for i, sp in enumerate(splitters):
        sp_id = str(uuid.uuid4())
        splitter_ids.append(sp_id)
        sp_pid = _normalize_pid(sp.get("probe_id"))
        sp_type = sp.get("type", "Dichroic")
        nodes.append(_node(
            id=sp_id, type_="splitter",
            title=f"{sp_type} Splitter {i + 1}",
            inputs=["In"], outputs=["Transmission", "Reflection"],
            config={"probe_id": sp_pid, "splitter_type": sp_type},
            pos=(550.0 + i * 30.0, 200.0 + i * 80.0),
        ))
        edges.append({"source": prev_node_id, "source_port": prev_port, "target": sp_id, "target_port": 0})

        # Transmission → detector i
        det = detectors[i] if i < len(detectors) else {}
        det_name = det.get("name", f"Channel {i + 1}")
        bp_pid = _normalize_pid(det.get("bandpass_probe_id"))
        dn_id = str(uuid.uuid4())

        chain_node = sp_id
        chain_port = 1  # Transmission output

        if bp_pid:
            bp_node_id = str(uuid.uuid4())
            nodes.append(_node(
                id=bp_node_id, type_="filter",
                title=f"Bandpass: {det_name}",
                inputs=["In"], outputs=["Out"],
                config={"probe_id": bp_pid},
                pos=(700.0 + i * 30.0, 100.0 + i * 200.0),
            ))
            edges.append({"source": chain_node, "source_port": chain_port, "target": bp_node_id, "target_port": 0})
            chain_node = bp_node_id
            chain_port = 1

        qe_pid = _normalize_pid(det.get("qe_probe_id"))
        nodes.append(_node(
            id=dn_id, type_="detector",
            title=det_name,
            inputs=["In"], outputs=[],
            config={"detector_name": det_name, "probe_id": qe_pid},
            pos=(850.0 + i * 30.0, 100.0 + i * 200.0),
        ))
        edges.append({"source": chain_node, "source_port": chain_port, "target": dn_id, "target_port": 0})

        # Next splitter feeds from this splitter's Reflection
        prev_node_id = sp_id
        prev_port = 2  # Reflection output

    # Last detector on the reflection port of the last splitter (or from ExciBW if no splitters)
    last_idx = len(splitters)
    if last_idx < len(detectors):
        det = detectors[last_idx]
        det_name = det.get("name", f"Channel {last_idx + 1}")
        bp_pid = _normalize_pid(det.get("bandpass_probe_id"))
        dn_id = str(uuid.uuid4())

        chain_node = prev_node_id
        chain_port = prev_port

        if bp_pid:
            bp_node_id = str(uuid.uuid4())
            nodes.append(_node(
                id=bp_node_id, type_="filter",
                title=f"Bandpass: {det_name}",
                inputs=["In"], outputs=["Out"],
                config={"probe_id": bp_pid},
                pos=(700.0 + last_idx * 30.0, 100.0 + last_idx * 200.0),
            ))
            edges.append({"source": chain_node, "source_port": chain_port, "target": bp_node_id, "target_port": 0})
            chain_node = bp_node_id
            chain_port = 1

        qe_pid = _normalize_pid(det.get("qe_probe_id"))
        nodes.append(_node(
            id=dn_id, type_="detector",
            title=det_name,
            inputs=["In"], outputs=[],
            config={"detector_name": det_name, "probe_id": qe_pid},
            pos=(850.0 + last_idx * 30.0, 100.0 + last_idx * 200.0),
        ))
        edges.append({"source": chain_node, "source_port": chain_port, "target": dn_id, "target_port": 0})

    # 5. Förster radius node
    kappa2 = config.get("kappa2", 0.6667)
    n_val = config.get("n", 1.33)
    nodes.append(_node(
        id=forster_id, type_="forster_radius", title="Förster Radius",
        inputs=[
            "Dye Data",
            {"name": "kappa2", "type": "number"},
            {"name": "n", "type": "number"},
        ],
        outputs=[],
        config={"kappa2": kappa2, "n": n_val, "_last_results": []},
        pos=(300.0, 500.0),
    ))
    edges.append({"source": sample_id, "source_port": 2, "target": forster_id, "target_port": 0})

    return {"nodes": nodes, "edges": edges, "version": 1}


# ---------------------------------------------------------------------------
# Preset I/O
# ---------------------------------------------------------------------------

def save_easy_preset(config: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_easy_preset(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_last_config(config: dict) -> None:
    try:
        EASY_LAST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EASY_LAST_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save last easy config: %s", exc)


def load_last_config() -> dict | None:
    try:
        if EASY_LAST_CONFIG_PATH.exists():
            with open(EASY_LAST_CONFIG_PATH) as f:
                return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load last easy config: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------


def _persist_fret_setting(key: str, value: float) -> None:
    """Persist a single fret setting to the user's YAML and in-memory dict."""
    cs_settings.setdefault("fret", {})[key] = value
    try:
        from chisurf.core.settings.path_utils import get_path
        import yaml
        settings_file = get_path("settings") / "settings_chisurf.yaml"
        data = {}
        try:
            with open(settings_file) as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            pass
        fret = data.setdefault("fret", {})
        fret[key] = value
        with open(settings_file, "w") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False)
    except Exception as exc:
        logger.warning("Failed to persist fret setting %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Sub-widgets
# ---------------------------------------------------------------------------


class _DyeTableWidget(QtWidgets.QWidget):
    """Filterable checkable dye table with QY/EC columns and spectra tooltips."""

    dyeSelectionChanged = QtCore.Signal()

    def __init__(self, probes: list[dict], db_path: str | None = None, parent=None):
        super().__init__(parent)
        self.probes = probes
        # PRD-23: construction is read-only — the MFDB adapter is opened lazily on
        # first tooltip render (see _ensure_db_adapter), never in __init__.
        self._db_path = db_path
        self._db_adapter: MFDatabaseAdapter | None = None
        self._db_opened = False
        self._setup_ui()
        self.populate()

    def _ensure_db_adapter(self) -> "MFDatabaseAdapter | None":
        """Open the MFDB adapter on first use (deferred from ``__init__``)."""
        if not self._db_opened:
            self._db_opened = True
            if self._db_path:
                try:
                    self._db_adapter = MFDatabaseAdapter(MFDatabase(self._db_path))
                except Exception:
                    self._db_adapter = None
        return self._db_adapter

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter dyes…")
        self.filter_edit.setFixedHeight(22)
        layout.addWidget(self.filter_edit)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Dye", "QY", "EC"])
        self.table.setStyleSheet("""
            QTableWidget { background: #1a1a1a; border: 1px solid #444; color: #eee; gridline-color: #333; font-size: 9px; }
            QHeaderView::section { background: #2a2a2a; padding: 1px; border: 1px solid #444; font-size: 8px; color: #999; }
            QTableWidget::item { padding: 0px; }
        """)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(1, 45)
        self.table.setColumnWidth(2, 55)
        vh = self.table.verticalHeader()
        vh.setVisible(False)
        vh.setDefaultSectionSize(16)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        layout.addWidget(self.table)

        self.filter_edit.textChanged.connect(self._on_filter)
        self.table.itemChanged.connect(self.dyeSelectionChanged.emit)

    def __del__(self):
        if self._db_adapter is not None:
            try:
                self._db_adapter.db.close()
            except Exception:
                pass

    def _on_filter(self):
        txt = self.filter_edit.text().lower()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                self.table.setRowHidden(row, txt not in item.text().lower())

    def _tooltip_html(self, probe_id: int) -> str:
        img = _render_spectra_thumbnail(probe_id, self._ensure_db_adapter())
        if not img:
            return ""
        name = ""
        for p in self.probes:
            if p.get("probe_id") == probe_id:
                name = p.get("name", "")
                break
        return f"<div style='text-align:center'><b>{name}</b><br>{img}</div>"

    def populate(self, selected_ids: list | None = None, dye_props: dict | None = None):
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        selected_ids = selected_ids or []
        dye_props = dye_props or {}
        valid = [(p["probe_id"], p["name"], p.get("qy", 1.0), p.get("ec", 1.0))
                 for p in self.probes if p.get("has_abs") and p.get("has_em")]
        self.table.setRowCount(len(valid))
        for row, (pid, name, dqy, dec) in enumerate(valid):
            ni = _SpectraTooltipItem(name, pid, render_fn=self._tooltip_html)
            ni.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            ni.setCheckState(QtCore.Qt.Checked if pid in selected_ids else QtCore.Qt.Unchecked)
            ni.setData(QtCore.Qt.UserRole, pid)
            self.table.setItem(row, 0, ni)
            ps = str(pid)
            qy = dye_props.get(ps, {}).get("qy", dqy)
            ec = dye_props.get(ps, {}).get("ec", dec)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(qy)))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(ec)))
        self.table.blockSignals(False)
        self.table.setUpdatesEnabled(True)

    def get_selected_dyes(self) -> dict[str, dict]:
        result = {}
        for row in range(self.table.rowCount()):
            ni = self.table.item(row, 0)
            if ni and ni.checkState() == QtCore.Qt.Checked:
                pid = ni.data(QtCore.Qt.UserRole)
                try:
                    qy = float(self.table.item(row, 1).text())
                except (ValueError, TypeError, AttributeError):
                    qy = 1.0
                try:
                    ec = float(self.table.item(row, 2).text())
                except (ValueError, TypeError, AttributeError):
                    ec = 1.0
                result[str(pid)] = {"qy": qy, "ec": ec}
        return result

    def set_selected_dyes(self, dyes: dict[str, dict]) -> None:
        self.populate(selected_ids=[_normalize_pid(p) for p in dyes], dye_props=dyes)

    def close_db(self):
        if self._db_adapter is not None:
            try:
                self._db_adapter.db.close()
            except Exception:
                pass
            self._db_adapter = None


class _EmissionSplitterTableWidget(QtWidgets.QWidget):
    """Table of cascaded emission splitters (dichroic or polarizer)."""

    changed = QtCore.Signal()

    def __init__(self, probes: list[dict], parent=None):
        super().__init__(parent)
        self.probes = probes
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Type", "Probe"])
        self.table.setStyleSheet("""
            QTableWidget { background: #1a1a1a; border: 1px solid #444; color: #eee; gridline-color: #333; font-size: 10px; }
            QHeaderView::section { background: #2a2a2a; padding: 2px; border: 1px solid #444; font-size: 8px; color: #999; }
        """)
        hh = self.table.horizontalHeader()
        for c in range(2):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)
        vh = self.table.verticalHeader()
        vh.setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setRowCount(0)

        self.add_btn = QtWidgets.QPushButton("+ Add Splitter")
        self.add_btn.clicked.connect(self._add_row)
        layout.addWidget(self.table)
        layout.addWidget(self.add_btn)

    def _add_row(self, type_name: str = "Dichroic", probe_id=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        type_cb = QtWidgets.QComboBox()
        type_cb.addItems(["Dichroic", "Polarizer"])
        idx = type_cb.findText(type_name, QtCore.Qt.MatchFixedString)
        if idx >= 0:
            type_cb.setCurrentIndex(idx)
        type_cb.currentIndexChanged.connect(self.changed.emit)
        self.table.setCellWidget(row, 0, type_cb)
        probe_cb = _make_combo(self.table, self.probes, "category:dichroic,polarizer")
        _set_combo(probe_cb, probe_id)
        probe_cb.currentIndexChanged.connect(self.changed.emit)
        self.table.setCellWidget(row, 1, probe_cb)

    def get_splitters(self) -> list[dict]:
        result = []
        for row in range(self.table.rowCount()):
            type_cb = self.table.cellWidget(row, 0)
            probe_cb = self.table.cellWidget(row, 1)
            if type_cb is None:
                continue
            result.append({
                "type": type_cb.currentText(),
                "probe_id": _combo_value(probe_cb),
            })
        return result

    def set_splitters(self, splitters: list[dict]):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for s in splitters:
            self._add_row(
                type_name=s.get("type", "Dichroic"),
                probe_id=_normalize_pid(s.get("probe_id")),
            )
        self.table.blockSignals(False)
        self.changed.emit()


class _DetectorTableWidget(QtWidgets.QWidget):
    """Table of detector channels, each with bandpass filter and QE probe."""

    changed = QtCore.Signal()

    def __init__(self, probes: list[dict], parent=None):
        super().__init__(parent)
        self.probes = probes
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Channel", "Bandpass", "QE"])
        self.table.setStyleSheet("""
            QTableWidget { background: #1a1a1a; border: 1px solid #444; color: #eee; gridline-color: #333; font-size: 10px; }
            QHeaderView::section { background: #2a2a2a; padding: 2px; border: 1px solid #444; font-size: 8px; color: #999; }
        """)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        vh = self.table.verticalHeader()
        vh.setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setRowCount(0)

        self.add_btn = QtWidgets.QPushButton("+ Add Channel")
        self.add_btn.clicked.connect(self._add_row)
        layout.addWidget(self.table)
        layout.addWidget(self.add_btn)

    def _add_row(self, name: str = "", bp_pid=None, qe_pid=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        if not name:
            parts = [r for r in range(self.table.rowCount())]
            name = f"Channel {len(parts)}"
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        bp = _make_combo(self.table, self.probes, "category:filter")
        qe = _make_combo(self.table, self.probes, "category:detector")
        _set_combo(bp, bp_pid)
        _set_combo(qe, qe_pid)
        bp.currentIndexChanged.connect(self.changed.emit)
        qe.currentIndexChanged.connect(self.changed.emit)
        self.table.setCellWidget(row, 1, bp)
        self.table.setCellWidget(row, 2, qe)
        self.table.itemChanged.connect(self.changed.emit)

    def set_detectors(self, detectors: list[dict]):
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(0)
        for d in detectors:
            self._add_row(
                name=d.get("name", ""),
                bp_pid=_normalize_pid(d.get("bandpass_probe_id")),
                qe_pid=_normalize_pid(d.get("qe_probe_id")),
            )
        self.table.blockSignals(False)
        self.table.setUpdatesEnabled(True)

    def get_detectors(self) -> list[dict]:
        result = []
        for row in range(self.table.rowCount()):
            ni = self.table.item(row, 0)
            if ni is None:
                continue
            name = ni.text().strip()
            if not name:
                continue
            result.append({
                "name": name,
                "bandpass_probe_id": _combo_value(self.table.cellWidget(row, 1)),
                "qe_probe_id": _combo_value(self.table.cellWidget(row, 2)),
            })
        return result

    def set_detector_names(self, names: list[str]):
        for i, n in enumerate(names):
            if i < self.table.rowCount():
                ni = self.table.item(i, 0)
                if ni:
                    ni.setText(n)


from chisurf.gui.widgets.collapsible_box import CollapsibleBox as _CollapsibleBox
from chisurf.gui.widgets.spectra_tooltip import (
    TooltipItem as _SpectraTooltipItem,
    render_spectra_thumbnail as _render_spectra_thumbnail,
)

# ---------------------------------------------------------------------------
# Easy mode widget — load optical path (full graph) and change filters/dyes
# ---------------------------------------------------------------------------

class LightPathEasyWidget(QtWidgets.QWidget):
    """Easy mode: load an optical path preset, then pick the spectra probes."""

    def __init__(
        self,
        probes: list[dict],
        parent=None,
        db_path: str | None = None,
    ):
        super().__init__(parent)
        self.probes = probes
        self._db_path = db_path
        self._last_results: dict | None = None
        self._suppress_recalc = False
        self._suppress_form_sync = False
        self._recalc_timer = QtCore.QTimer()
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.timeout.connect(self.recalculate)
        self._graph_sync_timer = QtCore.QTimer()
        self._graph_sync_timer.setSingleShot(True)
        self._graph_sync_timer.timeout.connect(self._sync_to_graph)
        self._component_rows: list[QtWidgets.QWidget] = []
        self._exci_table: _SingleProbeTable | None = None
        self._lasers_edit: QtWidgets.QLineEdit | None = None
        self._splitter_tables: list[_SingleProbeTable] = []
        self._detector_widgets: list[dict] = []  # [{bp_table, qe_table, name_item}]
        self._setup_ui()
        self._connect_signals()
        self._restore_last_config()
        self._connect_controls()

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(4)

        # ── Preset toolbar ──
        bar = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("💾")
        self.btn_save.setToolTip("Save preset")
        self.btn_load = QtWidgets.QPushButton("📂")
        self.btn_load.setToolTip("Load preset")
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setPlaceholderText("Optical Path Presets…")
        self.btn_edit = QtWidgets.QPushButton("✏️")
        self.btn_edit.setToolTip("Edit in Full Simulator")
        bar.addWidget(self.btn_save)
        bar.addWidget(self.btn_load)
        bar.addWidget(self.preset_combo, 1)
        bar.addWidget(self.btn_edit)
        main_layout.addLayout(bar)

        # ── Scrollable component form ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.form_container = QtWidgets.QWidget()
        # Use a plain VBoxLayout; _populate_form() fills it with _CollapsibleBox sections
        self.form_layout = QtWidgets.QVBoxLayout(self.form_container)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setSpacing(2)
        scroll.setWidget(self.form_container)
        main_layout.addWidget(scroll, 1)

        self._refresh_preset_list()

    # ── Dynamic form population ──

    def _clear_form(self):
        """Remove all dynamically added sections from the form layout."""
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._component_rows.clear()
        self._splitter_tables.clear()
        self._detector_widgets.clear()
        self._exci_table = None
        self._lasers_edit = None
        self._sec_sim = None

    def _populate_form(self, cfg: dict):
        """Build collapsible sections from a config dict, keeping topology fixed."""
        self._clear_form()

        # ── Section: Optical Components (lasers, dichroic, splitters) ──
        sec_optics = _CollapsibleBox("Optical Components", expanded=True)
        self._component_rows.append(sec_optics)

        # Lasers row
        self._lasers_edit = QtWidgets.QLineEdit(cfg.get("lasers", "488:1.0, 640:1.0"))
        self._lasers_edit.setPlaceholderText("e.g. 488:1.0, 561:0.5, 640:1.0")
        self._lasers_edit.editingFinished.connect(self._schedule_recalc)
        sec_optics.add_row("Lasers:", self._lasers_edit)

        # Excitation dichroic — table header acts as the label
        self._exci_table = _SingleProbeTable(self.probes, db_path=self._db_path, filter_key="category:dichroic", header_text="Exci. Dichroic")
        self._exci_table.set_selected_probe_id(cfg.get("excitation_dichroic_probe_id"))
        self._exci_table.changed.connect(self._schedule_recalc)
        sec_optics.add_widget(self._exci_table)

        # Emission splitters — each table header acts as the label
        splitters = cfg.get("emission_splitters", [])
        if not splitters:
            legacy_pid = _normalize_pid(cfg.get("emission_splitter_probe_id"))
            legacy_type = cfg.get("emission_splitter_type", "Dichroic")
            if legacy_pid is not None:
                splitters = [{"type": legacy_type, "probe_id": legacy_pid}]

        self._splitter_tables = []
        for i, sp in enumerate(splitters):
            sp_type = sp.get("type", "Dichroic")
            tbl = _SingleProbeTable(self.probes, db_path=self._db_path, filter_key="category:dichroic,polarizer", header_text=f"Splitter {i + 1}")
            tbl._splitter_type = sp_type
            tbl.table.setProperty("_splitter_type", sp_type)
            tbl.set_selected_probe_id(sp.get("probe_id"))
            tbl.changed.connect(self._schedule_recalc)
            sec_optics.add_widget(tbl)
            self._splitter_tables.append(tbl)

        # Placeholder splitter rows when detectors outnumber splitters
        detectors = cfg.get("detectors", [])
        n_det = max(len(splitters) + 1, len(detectors))
        while len(detectors) < n_det:
            detectors.append({"name": f"Channel {len(detectors) + 1}"})

        n_missing = n_det - 1 - len(splitters)
        splitter_idx = len(splitters) + 1
        for i in range(n_missing):
            tbl = _SingleProbeTable(self.probes, db_path=self._db_path, filter_key="category:dichroic,polarizer", header_text=f"Splitter {splitter_idx}")
            tbl._splitter_type = "Dichroic"
            tbl.table.setProperty("_splitter_type", "Dichroic")
            tbl.changed.connect(self._schedule_recalc)
            sec_optics.add_widget(tbl)
            self._splitter_tables.append(tbl)
            splitter_idx += 1

        self.form_layout.addWidget(sec_optics)

        # ── Section: Detectors ──
        sec_channels = _CollapsibleBox("Detectors", expanded=True)
        self._component_rows.append(sec_channels)

        self._detector_widgets = []
        is_polarizer = _is_polarizer_template(detectors[:n_det])
        det_grid_w = QtWidgets.QWidget()
        det_grid_w.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        det_grid = QtWidgets.QGridLayout(det_grid_w)
        det_grid.setContentsMargins(0, 0, 0, 0)
        det_grid.setSpacing(2)
        det_grid.setColumnStretch(0, 1)
        det_grid.setColumnStretch(1, 1)

        # Column headers row (row 0)
        _hdr_style = "color: #9ba3af; font-size: 9px; font-weight: bold; padding: 1px 4px;"
        for col_idx, col_label in enumerate(["Bandpass", "QE"]):
            lbl = QtWidgets.QLabel(col_label)
            lbl.setStyleSheet(_hdr_style)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            det_grid.addWidget(lbl, 0, col_idx)
        det_grid.setRowStretch(0, 0)

        for i, det in enumerate(detectors[:n_det]):
            det_name = det.get("name", f"Channel {i + 1}")
            if is_polarizer:
                color_idx, pol = _split_detector_name(det_name)
                if color_idx == 0:
                    color_idx = i // 2 + 1
                if pol:
                    label = f"C{color_idx}&thinsp;<sub>{pol}</sub>"
                else:
                    label = f"C{color_idx}"
            else:
                label = f"C{i + 1}"

            bp_tbl = _SingleProbeTable(self.probes, db_path=self._db_path,
                                       filter_key="category:filter", show_header=True, header_text=label)
            bp_tbl.set_selected_probe_id(det.get("bandpass_probe_id"))
            bp_tbl.changed.connect(self._schedule_recalc)
            det_grid.addWidget(bp_tbl, i + 1, 0)

            qe_tbl = _SingleProbeTable(self.probes, db_path=self._db_path,
                                       filter_key="category:detector", show_header=True, header_text="QE")
            qe_tbl.set_selected_probe_id(det.get("qe_probe_id"))
            qe_tbl.changed.connect(self._schedule_recalc)
            det_grid.addWidget(qe_tbl, i + 1, 1)

            det_grid.setRowStretch(i + 1, 1)

            self._detector_widgets.append({"bp": bp_tbl, "qe": qe_tbl, "name": det_name})

        sec_channels.add_widget(det_grid_w)
        self.form_layout.addWidget(sec_channels)

        # ── Section: Fluorophores ──
        sec_dyes = _CollapsibleBox("Fluorophores", expanded=True)
        self._component_rows.append(sec_dyes)
        self.dye_table = _DyeTableWidget(self.probes, db_path=self._db_path)
        dyes = cfg.get("dyes", {})
        if dyes:
            self.dye_table.set_selected_dyes(dyes)
        self.dye_table.dyeSelectionChanged.connect(self._schedule_recalc)
        sec_dyes.add_widget(self.dye_table)
        self.form_layout.addWidget(sec_dyes)

        # ── Section: Simulation (controls, parameters, results) ──
        sec_sim = _CollapsibleBox("Simulation", expanded=True)
        self._sec_sim = sec_sim
        self._component_rows.append(sec_sim)

        fret_cfg = cs_settings.get("fret", {})
        kappa2_default = fret_cfg.get("kappa2", 0.6667)
        n_default = fret_cfg.get("n", 1.33)

        self.auto_recalc_cb = QtWidgets.QCheckBox("Auto recalculate")
        self.auto_recalc_cb.setChecked(True)
        self.btn_calc = QtWidgets.QPushButton("Recalculate")
        self.btn_calc.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self.kappa2_spin = QtWidgets.QDoubleSpinBox()
        self.kappa2_spin.setRange(0, 4)
        self.kappa2_spin.setSingleStep(0.1)
        self.kappa2_spin.setValue(cfg.get("kappa2", kappa2_default))
        self.kappa2_spin.valueChanged.connect(lambda v: _persist_fret_setting("kappa2", v))
        self.kappa2_spin.valueChanged.connect(self._schedule_recalc)
        self.n_spin = QtWidgets.QDoubleSpinBox()
        self.n_spin.setRange(1.0, 2.0)
        self.n_spin.setSingleStep(0.01)
        self.n_spin.setValue(cfg.get("n", n_default))
        self.n_spin.valueChanged.connect(lambda v: _persist_fret_setting("n", v))
        self.n_spin.valueChanged.connect(self._schedule_recalc)

        sim_row = QtWidgets.QHBoxLayout()
        sim_row.setSpacing(4)
        k2_lbl = QtWidgets.QLabel("kappa²:")
        k2_lbl.setStyleSheet("color: #9ba3af; font-size: 10px;")
        n_lbl = QtWidgets.QLabel("n:")
        n_lbl.setStyleSheet("color: #9ba3af; font-size: 10px;")
        sim_row.addWidget(k2_lbl)
        sim_row.addWidget(self.kappa2_spin)
        sim_row.addWidget(n_lbl)
        sim_row.addWidget(self.n_spin)
        sim_row.addStretch()
        sim_row.addWidget(self.auto_recalc_cb)
        sim_row.addWidget(self.btn_calc)
        sim_widget = QtWidgets.QWidget()
        sim_widget.setLayout(sim_row)
        sec_sim.add_widget(sim_widget)

        self.results_tabs = QtWidgets.QTabWidget()
        _result_tbl_style = """
            QTableWidget { background: #1a1a1a; border: 1px solid #444; color: #eee; gridline-color: #333; font-size: 10px; }
            QHeaderView::section { background: #2a2a2a; padding: 1px; border: 1px solid #444; font-size: 9px; color: #999; }
        """
        self.forster_table = QtWidgets.QTableWidget()
        self.forster_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.forster_table.setStyleSheet(_result_tbl_style)
        self.forster_table.verticalHeader().setDefaultSectionSize(20)
        self.ex_table = QtWidgets.QTableWidget()
        self.ex_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ex_table.setStyleSheet(_result_tbl_style)
        self.ex_table.verticalHeader().setDefaultSectionSize(20)
        self.em_table = QtWidgets.QTableWidget()
        self.em_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.em_table.setStyleSheet(_result_tbl_style)
        self.em_table.verticalHeader().setDefaultSectionSize(20)
        self.det_table = QtWidgets.QTableWidget()
        self.det_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.det_table.setStyleSheet(_result_tbl_style)
        self.det_table.verticalHeader().setDefaultSectionSize(20)
        self.results_tabs.addTab(self.forster_table, "Förster R₀ [Å]")
        self.results_tabs.addTab(self.ex_table, "Excitation CT")
        self.results_tabs.addTab(self.em_table, "Emission CT")
        self.results_tabs.addTab(self.det_table, "Detected CT")
        self.results_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sec_sim.add_widget(self.results_tabs, 1)

        self.form_layout.addWidget(sec_sim, 1)

        # Dynamically surrender/claim vertical space when Simulation is collapsed/expanded
        sec_sim.toggled.connect(self._on_sec_sim_toggled)

    # ── Signals ──

    def _connect_signals(self):
        self.btn_save.clicked.connect(self._on_save_preset)
        self.btn_load.clicked.connect(self._on_load_preset)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self.btn_edit.clicked.connect(self._on_edit)

    def _connect_controls(self):
        """Connect signals for controls created by _populate_form."""
        if hasattr(self, 'btn_calc') and self.btn_calc is not None:
            self.btn_calc.clicked.connect(self.recalculate)
        if hasattr(self, 'auto_recalc_cb') and self.auto_recalc_cb is not None:
            self.auto_recalc_cb.toggled.connect(self._on_auto_recalc)

    def _rebuild_graph(self, cfg: dict | None = None) -> None:
        """Replace the parent Full Simulator graph with a fresh one from *cfg*."""
        if cfg is None:
            cfg = self._get_config()
        graph = build_easy_graph(cfg)
        from chisurf.plugins.core.lightpath_simulator.gui.tool import LightPathSimulatorWidget
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, LightPathSimulatorWidget):
            parent = parent.parentWidget()
        if parent is not None:
            parent._is_syncing_easy = True
            try:
                parent.load_graph_from_dict(graph)
            finally:
                parent._is_syncing_easy = False

    # ── Preset management ──

    def _preset_path(self, name: str) -> Path:
        return OPTICAL_PRESETS_DIR / f"{name.strip()}.json"

    def _refresh_preset_list(self):
        self.preset_combo.blockSignals(True)
        current = self.preset_combo.currentText()
        self.preset_combo.clear()
        for template in load_template_library():
            self.preset_combo.addItem(
                f"Template: {template.get('name', template.get('id', 'Untitled'))}",
                {
                    "kind": "template",
                    "id": template.get("id"),
                    "path": template.get("_path"),
                    "config": copy.deepcopy(template.get("config", {})),
                },
            )
        if OPTICAL_PRESETS_DIR.exists():
            if self.preset_combo.count() > 0:
                self.preset_combo.insertSeparator(self.preset_combo.count())
            for f in sorted(OPTICAL_PRESETS_DIR.glob("*.json")):
                self.preset_combo.addItem(
                    f"Preset: {f.stem}",
                    {"kind": "preset", "path": str(f)},
                )
        idx = self.preset_combo.findText(current)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, idx: int):
        item = self.preset_combo.itemData(idx)
        if not isinstance(item, dict):
            return
        loaded = False
        try:
            self._suppress_recalc = True
            self._suppress_form_sync = True
            if item.get("kind") == "template":
                cfg = copy.deepcopy(item.get("config", {}))
            else:
                path = Path(str(item.get("path", "")))
                if not path.exists():
                    return
                cfg = load_easy_preset(str(path))
            # Detect graph-dict format (saved from Full Simulator) → convert
            if "nodes" in cfg:
                cfg = _graph_to_config(cfg)
            self._populate_form(cfg)
            self._connect_controls()
            self._rebuild_graph(cfg)
            loaded = True
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Failed", str(exc))
        finally:
            self._suppress_recalc = False
            self._suppress_form_sync = False
        if loaded:
            self._schedule_recalc()

    # ── Save / Load presets ──

    def _on_save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name:
            return
        EASY_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        path = EASY_PRESETS_DIR / f"{name.strip()}.json"
        cfg = self._get_config()
        save_easy_preset(cfg, path)
        self._refresh_preset_list()

    def _on_load_preset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Preset", str(EASY_PRESETS_DIR), "JSON (*.json)"
        )
        if not path:
            return
        try:
            cfg = load_easy_preset(path)
            self._suppress_recalc = True
            self._suppress_form_sync = True
            self._populate_form(cfg)
            self._connect_controls()
            self._rebuild_graph(cfg)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Failed", str(exc))
        finally:
            self._suppress_recalc = False
            self._suppress_form_sync = False
        self._schedule_recalc()

    # ── Edit → opens Full Simulator ──

    def _on_edit(self):
        from chisurf.plugins.core.lightpath_simulator.gui.tool import LightPathSimulatorWidget
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, LightPathSimulatorWidget):
            parent = parent.parentWidget()
        if parent is not None:
            for i in range(parent.dock_area.count()):
                if parent.dock_area.widget(i).objectName() == "lightpathGraphDock":
                    parent.dock_area.setCurrentIndex(i)
                    break
        else:
            w = LightPathSimulatorWidget()
            w.show()

    # ── Auto recalculate ──

    def _on_auto_recalc(self, checked: bool):
        if checked:
            self._schedule_recalc()

    def _on_sec_sim_toggled(self, expanded: bool):
        idx = self.form_layout.indexOf(self._sec_sim)
        if idx >= 0:
            self.form_layout.setStretch(idx, 1 if expanded else 0)

    def _schedule_recalc(self):
        if self._suppress_recalc:
            return
        if self.auto_recalc_cb.isChecked():
            self._recalc_timer.start(300)
        if not self._suppress_form_sync:
            self._graph_sync_timer.start(400)

    def _sync_to_graph(self):
        """Push current Easy Mode config into the parent Full Simulator, rebuilding the graph."""
        if self._suppress_form_sync:
            return
        self._rebuild_graph()

    # ── Config ──

    def _get_config(self) -> dict:
        lasers = self._lasers_edit.text() if self._lasers_edit else "488:1.0, 640:1.0"
        exci_pid = self._exci_table.get_selected_probe_id() if self._exci_table else None
        splitters = []
        for tbl in self._splitter_tables:
            spl_meta = getattr(tbl, "_splitter_type", "Dichroic") or "Dichroic"
            splitters.append({"type": spl_meta, "probe_id": tbl.get_selected_probe_id()})
        detectors = []
        for dw in self._detector_widgets:
            detectors.append({
                "name": dw["name"],
                "bandpass_probe_id": dw["bp"].get_selected_probe_id(),
                "qe_probe_id": dw["qe"].get_selected_probe_id(),
            })
        splitters = _clean_splitters_for_detector_count(splitters, len(detectors))
        dyes = self.dye_table.get_selected_dyes()
        kappa2 = self.kappa2_spin.value() if hasattr(self, "kappa2_spin") else 0.6667
        n_val = self.n_spin.value() if hasattr(self, "n_spin") else 1.33
        return {
            "lasers": lasers,
            "excitation_dichroic_probe_id": exci_pid,
            "emission_splitters": splitters,
            "detectors": detectors,
            "dyes": dyes,
            "kappa2": kappa2,
            "n": n_val,
        }

    # ── Simulation ──

    def recalculate(self):
        cfg = self._get_config()
        if not cfg.get("dyes"):
            QtWidgets.QMessageBox.warning(self, "No Dyes", "Select at least one dye.")
            return
        try:
            graph = build_easy_graph(cfg)
            with MFDatabase(resolve_db_path()) as db:
                result = _simulate_with_db(graph, db)
            self._last_results = result
            self._show_results(result)
            self._save_last_config()
        except Exception as exc:
            logger.error("Easy mode simulation failed: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Simulation Failed", str(exc))

    def _save_last_config(self):
        try:
            EASY_LAST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            save_easy_preset(self._get_config(), str(EASY_LAST_CONFIG_PATH))
        except Exception as exc:
            logger.warning("Failed to save last config: %s", exc)

    def _restore_last_config(self):
        try:
            if EASY_LAST_CONFIG_PATH.exists():
                cfg = load_easy_preset(str(EASY_LAST_CONFIG_PATH))
                if "nodes" in cfg:
                    cfg = _graph_to_config(cfg)
                self._suppress_recalc = True
                self._suppress_form_sync = True
                self._populate_form(cfg)
                self._suppress_recalc = False
                self._suppress_form_sync = False
        except Exception as exc:
            logger.warning("Failed to restore last config: %s", exc)

    def _show_results(self, result: dict):
        fr = self._extract_forster(result)
        self._fill_table(self.forster_table, fr)
        cm = result.get("crosstalk_matrices", {})
        self._fill_table(self.ex_table, cm.get("excitation", {}))
        self._fill_table(self.em_table, cm.get("emission", {}))
        self._fill_table(self.det_table, cm.get("detected", {}))

    @staticmethod
    def _extract_forster(result: dict) -> dict:
        for ns in result.get("states", {}).values():
            rows = ns.get("config", {}).get("_last_results", [])
            if rows:
                donors = sorted({r["donor"] for r in rows})
                acceptors = sorted({r["acceptor"] for r in rows})
                mp = {(r["donor"], r["acceptor"]): r["r0"] for r in rows}
                vals = [[mp.get((d, a), 0.0) for a in acceptors] for d in donors]
                return {"rows": donors, "columns": acceptors, "values": vals}
        return {"rows": [], "columns": [], "values": []}

    @staticmethod
    def _fill_table(table: QtWidgets.QTableWidget, matrix: dict):
        rows = matrix.get("rows", [])
        cols = matrix.get("columns", [])
        vals = matrix.get("values", [])
        table.clear()
        table.setRowCount(len(rows))
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels([str(c) for c in cols])
        table.setVerticalHeaderLabels([str(r) for r in rows])
        table.verticalHeader().setVisible(len(rows) > 0)
        for ri, rv in enumerate(vals):
            for ci, v in enumerate(rv):
                txt = f"{float(v):.1f}" if isinstance(v, (int, float)) else str(v)
                item = QtWidgets.QTableWidgetItem(txt)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(ri, ci, item)

    def get_optical_config(self) -> dict:
        cfg = self._get_config()
        if self._last_results:
            cfg["_cached_results"] = {
                "forster": self._extract_forster(self._last_results),
                "crosstalk_matrices": self._last_results.get("crosstalk_matrices", {}),
            }
        return cfg

    def set_optical_config(self, config: dict):
        self._suppress_recalc = True
        self._suppress_form_sync = True
        self._populate_form(config)
        cached = config.get("_cached_results")
        if cached:
            self._fill_table(self.forster_table, cached.get("forster", {}))
            cm = cached.get("crosstalk_matrices", {})
            if cm:
                self._fill_table(self.ex_table, cm.get("excitation", {}))
                self._fill_table(self.em_table, cm.get("emission", {}))
                self._fill_table(self.det_table, cm.get("detected", {}))
        self._suppress_recalc = False
        self._suppress_form_sync = False


# ---------------------------------------------------------------------------
# Standalone dialog
# ---------------------------------------------------------------------------

class LightPathEasyDialog(QtWidgets.QDialog):
    """Dialog wrapping LightPathEasyWidget for use from Detector Wizard."""

    def __init__(self, probes: list[dict], parent=None,
                 detector_names: list[str] | None = None,
                 optical_config: dict | None = None,
                 db_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Optical Setup — Easy Mode")
        self.resize(620, 700)

        layout = QtWidgets.QVBoxLayout(self)
        self.easy_widget = LightPathEasyWidget(probes, self, db_path=db_path)
        layout.addWidget(self.easy_widget, 1)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        if optical_config:
            self.easy_widget.set_optical_config(optical_config)

    def get_optical_config(self) -> dict:
        return self.easy_widget.get_optical_config()
