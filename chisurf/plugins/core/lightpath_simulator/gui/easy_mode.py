"""Easy mode for the Light Path Simulator — form-based optical configuration."""

import copy
import json
import uuid
import logging
from pathlib import Path
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.plugins.core.lightpath_simulator.core.workflow import (
    _simulate_with_db,
    resolve_db_path,
    MFDatabaseAdapter,
)
from chisurf.core.mfdb.repository import MFDatabase

logger = logging.getLogger(__name__)

EASY_LAST_CONFIG_PATH = Path.home() / ".chisurf" / "settings" / "lightpath_easy_last.json"
EASY_PRESETS_DIR = Path.home() / ".chisurf" / "presets" / "lightpath_easy"
OPTICAL_PRESETS_DIR = Path.home() / ".chisurf" / "presets" / "lightpath_optical"
DYE_PRESETS_DIR = Path.home() / ".chisurf" / "presets" / "lightpath_dyes"


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
    combo.addItem("None", None)
    for p in probes:
        if p.get(filter_key):
            combo.addItem(p["name"], p["probe_id"])
    return combo


def _combo_value(combo):
    return combo.itemData(combo.currentIndex()) if combo else None


def _set_combo(combo, pid):
    if pid is not None and combo is not None:
        idx = combo.findData(_normalize_pid(pid))
        if idx >= 0:
            combo.setCurrentIndex(idx)


# ---------------------------------------------------------------------------
# Graph ↔ config conversion
# ---------------------------------------------------------------------------

def _graph_to_config(graph: dict) -> dict:
    """Convert a GraphDef graph dict back into an easy-mode config dict.

    This handles presets saved from the Full Simulator (which store the
    full node/edge graph) so the Easy Mode form can populate correctly.
    """
    nodes_list = graph.get("nodes", [])
    edges_list = graph.get("edges", [])
    nodes_map: dict[str, dict] = {n["id"]: n for n in nodes_list}

    # Build adjacency: source_id → [(source_port, target_id, target_port)]
    forward: dict[str, list[tuple[int, str, int]]] = {}
    for e in edges_list:
        src = e["source"]
        forward.setdefault(src, []).append(
            (e["source_port"], e["target"], e["target_port"])
        )

    config: dict = {}

    # --- Lasers ---
    for n in nodes_list:
        if n["type"] == "light_source":
            cfg = n.get("config", {})
            config["lasers"] = cfg.get("manual_lines", "488:1.0, 640:1.0")
            break

    # --- Excitation dichroic ---
    # Find the dichroic that sits between Light Source and Sample
    # (type="splitter" connected to light_source output port 0 → input port 0)
    for n in nodes_list:
        if n["type"] == "splitter":
            pid = n.get("config", {}).get("probe_id")
            if pid is not None:
                config["excitation_dichroic_probe_id"] = pid
                break

    # --- Emission splitters & Detectors ---
    # Walk the emission path: start from a splitter that is *not* connected
    # to a light source, follow its reflection port to cascaded splitters.
    splitters: list[dict] = []
    detectors: list[dict] = []

    # Find the first emission splitter: a splitter whose input comes from
    # another splitter's output (the ExciBW node).
    def node_type_of(nid: str) -> str:
        nd = nodes_map.get(nid)
        return nd["type"] if nd else ""

    first_ems_id: str | None = None
    for n in nodes_list:
        if n["type"] != "splitter":
            continue
        nid = n["id"]
        # Check incoming edges
        for src_id, out_edges in forward.items():
            for sp, tgt, tp in out_edges:
                if tgt == nid and tp == 0:
                    src_type = node_type_of(src_id)
                    # If the source is also a splitter or a sample, this
                    # is in the emission path (not fed by light_source).
                    if src_type in ("splitter", "sample"):
                        first_ems_id = nid
                        break
            if first_ems_id is not None:
                break
        if first_ems_id is not None:
            break

    if first_ems_id is None and len(splitters) == 0:
        # Fallback: assume all splitters except the first are emission splitters
        found_exci = False
        for n in nodes_list:
            if n["type"] == "splitter":
                pid = n.get("config", {}).get("probe_id")
                if not found_exci:
                    found_exci = True
                    continue
                if pid is not None:
                    splitters.append({
                        "type": n.get("config", {}).get("splitter_type", "Dichroic"),
                        "probe_id": pid,
                    })

    if first_ems_id is not None:
        # Walk the cascade: follow reflection ports
        cur_id = first_ems_id
        visited: set[str] = set()
        while cur_id and cur_id not in visited:
            visited.add(cur_id)
            nd = nodes_map[cur_id]
            cfg = nd.get("config", {})
            sp_type = cfg.get("splitter_type", "Dichroic")
            splitters.append({
                "type": sp_type,
                "probe_id": cfg.get("probe_id"),
            })
            # Find detector on transmission port (port 1)
            for sp, tgt, tp in forward.get(cur_id, []):
                if sp == 1:
                    # Traverse to find detector: optional bandpass filter → detector
                    _det_name, _bp_pid, _qe_pid = _resolve_detector_chain(
                        tgt, nodes_map, forward
                    )
                    detectors.append({
                        "name": _det_name,
                        "bandpass_probe_id": _bp_pid,
                        "qe_probe_id": _qe_pid,
                    })

            # Find next splitter on reflection port (port 2)
            next_id = None
            for sp, tgt, tp in forward.get(cur_id, []):
                if sp == 2 and node_type_of(tgt) == "splitter":
                    next_id = tgt
                    break
            cur_id = next_id

        # Last detector on the reflection port of the final splitter
        if cur_id is None and visited:
            last_id = list(visited)[-1]
            for sp, tgt, tp in forward.get(last_id, []):
                if sp == 2:
                    _det_name, _bp_pid, _qe_pid = _resolve_detector_chain(
                        tgt, nodes_map, forward
                    )
                    detectors.append({
                        "name": _det_name,
                        "bandpass_probe_id": _bp_pid,
                        "qe_probe_id": _qe_pid,
                    })

    # Fallback: collect detectors directly from the node list
    if not detectors:
        for n in nodes_list:
            if n["type"] == "detector":
                det_cfg = n.get("config", {})
                detectors.append({
                    "name": det_cfg.get("detector_name", "Detector"),
                    "bandpass_probe_id": None,
                    "qe_probe_id": det_cfg.get("probe_id"),
                })

    config["emission_splitters"] = splitters
    config["detectors"] = detectors

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

    return config


def _resolve_detector_chain(
    start_id: str,
    nodes_map: dict[str, dict],
    forward: dict[str, list[tuple[int, str, int]]],
) -> tuple[str, Any, Any]:
    """Walk from a node to find the detector, its bandpass, and QE probe."""
    from qtpy import QtCore

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
            det_name = cfg.get("detector_name", "Detector")
            qe_pid = cfg.get("probe_id")
            break

        # Follow the single outgoing edge
        next_id = None
        for sp, tgt, tp in forward.get(cur, []):
            # Skip port 0 for filters if we already found a BP
            if ntype == "filter" and sp == 0:
                continue
            next_id = tgt
            break
        cur = next_id

    return det_name, bp_pid, qe_pid


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
      Light Source → Excitation Dichroic (reflection) → Sample
      Sample → Excitation Dichroic (transmission) → cascaded Emission Splitters → N Detectors
      Sample → Förster Radius

    Supports two config formats:
    - New: ``emission_splitters`` (list of ``{type, probe_id}``)
    - Legacy: ``emission_splitter_probe_id`` + ``emission_splitter_type`` (single splitter)
    """
    nodes = []
    edges = []

    light_id = str(uuid.uuid4())
    sample_id = str(uuid.uuid4())
    exci_fw_id = str(uuid.uuid4())
    exci_bw_id = str(uuid.uuid4())
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

    # 3. Excitation dichroic — forward path (laser → sample)
    exci_pid = _normalize_pid(config.get("excitation_dichroic_probe_id"))
    nodes.append(_node(
        id=exci_fw_id, type_="splitter",
        title="Exci. Dichroic (FW)",
        inputs=["In"], outputs=["Transmission", "Reflection"],
        config={"probe_id": exci_pid},
        pos=(175.0, 50.0),
    ))
    edges.append({"source": light_id, "source_port": 0, "target": exci_fw_id, "target_port": 0})
    edges.append({"source": exci_fw_id, "source_port": 2, "target": sample_id, "target_port": 0})

    # 4. Excitation dichroic — backward path (sample → emission splitters)
    nodes.append(_node(
        id=exci_bw_id, type_="splitter",
        title="Exci. Dichroic (BW)",
        inputs=["In"], outputs=["Transmission", "Reflection"],
        config={"probe_id": exci_pid},
        pos=(175.0, 350.0),
    ))
    edges.append({"source": sample_id, "source_port": 1, "target": exci_bw_id, "target_port": 0})

    # 5. Emission splitters (cascaded) → Detector channels
    splitters = config.get("emission_splitters", [])
    if not splitters:
        # Legacy: single splitter from emission_splitter_probe_id + emission_splitter_type
        legacy_pid = _normalize_pid(config.get("emission_splitter_probe_id"))
        legacy_type = config.get("emission_splitter_type", "Dichroic")
        if legacy_pid is not None:
            splitters = [{"type": legacy_type, "probe_id": legacy_pid}]

    detectors = config.get("detectors", [])
    n_detectors = len(splitters) + 1  # N splitters → N+1 detectors

    # Auto-generate splitters if more detectors than splitters allow
    while n_detectors < len(detectors):
        splitters.append({"type": "Dichroic", "probe_id": None})
        n_detectors = len(splitters) + 1

    # Ensure detector list has enough entries
    while len(detectors) < n_detectors:
        detectors.append({"name": f"Channel {len(detectors) + 1}"})

    # Build cascaded splitters
    prev_node_id = exci_bw_id
    prev_port = 1  # ExciBW Transmission output
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

    # 6. Förster radius node
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
# Sub-widgets
# ---------------------------------------------------------------------------

class _SpectraTooltipItem(QtWidgets.QTableWidgetItem):
    """QLabel-style item that lazily renders spectra tooltip on first hover."""

    def __init__(self, text: str, probe_id: int, render_fn=None):
        super().__init__(text)
        self._probe_id = probe_id
        self._render_fn = render_fn
        self._cached_html = None

    def data(self, role):
        if role == QtCore.Qt.ToolTipRole:
            if self._cached_html is None and self._render_fn is not None:
                self._cached_html = self._render_fn(self._probe_id) or ""
            return self._cached_html
        return super().data(role)


def _render_spectra_thumbnail(probe_id: int, adapter: MFDatabaseAdapter | None) -> str:
    """Render abs/em spectra as a small PNG embedded in an HTML img tag."""
    if adapter is None:
        return ""
    abs_spec = adapter.get_probe_spectrum(probe_id, "absorption")
    em_spec = adapter.get_probe_spectrum(probe_id, "emission")
    if not abs_spec and not em_spec:
        return ""

    w, h = 300, 130
    pm = QtGui.QPixmap(w, h)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)

    ml, mr, mt, mb = 10, 10, 5, 16
    pw = w - ml - mr
    ph = h - mt - mb

    all_wl = []
    if abs_spec:
        all_wl.extend(abs_spec[0])
    if em_spec:
        all_wl.extend(em_spec[0])
    if not all_wl:
        p.end()
        return ""

    x_min, x_max = min(all_wl), max(all_wl)
    x_range = x_max - x_min or 1

    def to_px(wl):
        return ml + (wl - x_min) / x_range * pw

    pen = QtGui.QPen(QtGui.QColor("#555"))
    p.setPen(pen)
    p.drawLine(ml, mt, ml, h - mb)
    p.drawLine(ml, h - mb, w - mr, h - mb)

    if abs_spec:
        pen = QtGui.QPen(QtGui.QColor("#4488ff"), 1.5)
        p.setPen(pen)
        wl, vals = abs_spec
        vmax = max(vals) if max(vals) > 0 else 1
        for i in range(len(wl) - 1):
            p.drawLine(
                int(to_px(wl[i])), int(h - mb - (vals[i] / vmax) * ph),
                int(to_px(wl[i + 1])), int(h - mb - (vals[i + 1] / vmax) * ph),
            )

    if em_spec:
        pen = QtGui.QPen(QtGui.QColor("#ff4444"), 1.5)
        p.setPen(pen)
        wl, vals = em_spec
        vmax = max(vals) if max(vals) > 0 else 1
        for i in range(len(wl) - 1):
            p.drawLine(
                int(to_px(wl[i])), int(h - mb - (vals[i] / vmax) * ph),
                int(to_px(wl[i + 1])), int(h - mb - (vals[i + 1] / vmax) * ph),
            )

    # X-axis tick labels
    tick_step = 50
    tick_start = ((int(x_min) + tick_step - 1) // tick_step) * tick_step
    fnt = p.font()
    fnt.setPointSize(7)
    p.setFont(fnt)
    pen = QtGui.QPen(QtGui.QColor("#aaa"))
    p.setPen(pen)
    for wl in range(tick_start, int(x_max) + 1, tick_step):
        if wl < x_min or wl > x_max:
            continue
        x = int(to_px(wl))
        p.drawLine(x, h - mb, x, h - mb + 3)
        txt = str(wl)
        text_rect = p.boundingRect(QtCore.QRect(0, 0, 0, 0), QtCore.Qt.AlignCenter, txt)
        p.drawText(x - text_rect.width() // 2, h - 2, txt)

    p.end()

    ba = QtCore.QByteArray()
    buf = QtCore.QBuffer(ba)
    buf.open(QtCore.QIODevice.WriteOnly)
    pm.save(buf, "PNG")
    buf.close()
    b64 = ba.toBase64().data().decode()
    return f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}">'


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

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter dyes…")
        layout.addWidget(self.filter_edit)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Dye", "QY", "EC"])
        self.table.setStyleSheet("""
            QTableWidget { background: #1a1a1a; border: 1px solid #444; color: #eee; gridline-color: #333; font-size: 10px; }
            QHeaderView::section { background: #2a2a2a; padding: 2px; border: 1px solid #444; font-size: 8px; color: #999; }
            QTableWidget::item { padding: 1px; }
        """)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 80)
        vh = self.table.verticalHeader()
        vh.setVisible(False)
        vh.setDefaultSectionSize(18)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
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
        probe_cb = _make_combo(self.table, self.probes, "has_trans")
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
        bp = _make_combo(self.table, self.probes, "has_trans")
        qe = _make_combo(self.table, self.probes, "has_qe")
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


# ---------------------------------------------------------------------------
# Easy mode widget — load optical path (full graph) and change filters/dyes
# ---------------------------------------------------------------------------

class LightPathEasyWidget(QtWidgets.QWidget):
    """Easy mode: load an optical path preset, then pick the spectra probes."""

    def __init__(self, probes: list[dict], parent=None, db_path: str | None = None):
        super().__init__(parent)
        self.probes = probes
        self._db_path = db_path
        self._last_results: dict | None = None
        self._suppress_recalc = False
        self._recalc_timer = QtCore.QTimer()
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.timeout.connect(self.recalculate)
        self._component_rows: list[QtWidgets.QWidget] = []
        self._exci_combo: QtWidgets.QComboBox | None = None
        self._lasers_edit: QtWidgets.QLineEdit | None = None
        self._splitter_widgets: list[QtWidgets.QComboBox] = []
        self._detector_widgets: list[dict] = []  # [{bp_combo, qe_combo, name_item}]
        self._setup_ui()
        self._connect_signals()
        self._restore_last_config()

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(4)

        # ── Preset toolbar ──
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Optical Path:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setPlaceholderText("Select preset…")
        self.btn_edit = QtWidgets.QPushButton("Edit…")
        bar.addWidget(self.preset_combo, 1)
        bar.addWidget(self.btn_edit)
        main_layout.addLayout(bar)

        # ── Scrollable component form ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.form_container = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QFormLayout(self.form_container)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setSpacing(4)
        self.form_layout.setHorizontalSpacing(8)
        scroll.setWidget(self.form_container)
        main_layout.addWidget(scroll, 1)

        # ── Auto-recalc ──
        self.auto_recalc_cb = QtWidgets.QCheckBox("Auto recalculate")
        self.auto_recalc_cb.setChecked(True)
        main_layout.addWidget(self.auto_recalc_cb)

        # ── Buttons ──
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("Recalculate")
        self.btn_calc.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.btn_full = QtWidgets.QPushButton("Open in Full Simulator")
        btn_row.addWidget(self.btn_calc)
        btn_row.addWidget(self.btn_full)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

        # ── Results ──
        self.results_group = QtWidgets.QGroupBox("Simulation Results")
        rl = QtWidgets.QVBoxLayout(self.results_group)
        self.results_tabs = QtWidgets.QTabWidget()
        self.forster_table = QtWidgets.QTableWidget()
        self.forster_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ex_table = QtWidgets.QTableWidget()
        self.ex_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.em_table = QtWidgets.QTableWidget()
        self.em_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.det_table = QtWidgets.QTableWidget()
        self.det_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.forster_table, "Förster R₀ [Å]")
        self.results_tabs.addTab(self.ex_table, "Excitation CT")
        self.results_tabs.addTab(self.em_table, "Emission CT")
        self.results_tabs.addTab(self.det_table, "Detected CT")
        rl.addWidget(self.results_tabs)
        self.results_group.setVisible(False)
        main_layout.addWidget(self.results_group, 0)

        self._refresh_preset_list()

    # ── Dynamic form population ──

    def _clear_form(self):
        """Remove all dynamically added rows from the form."""
        for w in self._component_rows:
            w.setParent(None)
            w.deleteLater()
        self._component_rows.clear()
        self._splitter_widgets.clear()
        self._detector_widgets.clear()
        self._exci_combo = None
        self._lasers_edit = None

    def _populate_form(self, cfg: dict):
        """Build form rows from a config dict, keeping topology fixed."""
        self._clear_form()

        # Lasers
        self._lasers_edit = QtWidgets.QLineEdit(cfg.get("lasers", "488:1.0, 640:1.0"))
        self._lasers_edit.setPlaceholderText("e.g. 488:1.0, 561:0.5, 640:1.0")
        self.form_layout.addRow("Lasers:", self._lasers_edit)
        self._component_rows.append(self._lasers_edit)
        self._lasers_edit.editingFinished.connect(self._schedule_recalc)

        # Excitation dichroic
        self._exci_combo = _make_combo(self, self.probes, "has_trans")
        _set_combo(self._exci_combo, cfg.get("excitation_dichroic_probe_id"))
        self.form_layout.addRow("Exci. Dichroic:", self._exci_combo)
        self._component_rows.append(self._exci_combo)
        self._exci_combo.currentIndexChanged.connect(self._schedule_recalc)

        # Emission splitters (read-only topology, changeable probes)
        splitters = cfg.get("emission_splitters", [])
        if not splitters:
            legacy_pid = _normalize_pid(cfg.get("emission_splitter_probe_id"))
            legacy_type = cfg.get("emission_splitter_type", "Dichroic")
            if legacy_pid is not None:
                splitters = [{"type": legacy_type, "probe_id": legacy_pid}]

        self._splitter_widgets = []
        for i, sp in enumerate(splitters):
            sp_type = sp.get("type", "Dichroic")
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(f"{sp_type}:")
            lbl.setStyleSheet("color: #aaa; font-size: 9px;")
            row.addWidget(lbl)
            cb = _make_combo(self, self.probes, "has_trans")
            _set_combo(cb, sp.get("probe_id"))
            cb.currentIndexChanged.connect(self._schedule_recalc)
            row.addWidget(cb, 1)
            container = QtWidgets.QWidget()
            container.setLayout(row)
            self.form_layout.addRow(f"Splitter {i + 1}:", container)
            self._component_rows.append(container)
            self._splitter_widgets.append(cb)

        # Detectors (read-only count, changeable BP + QE)
        detectors = cfg.get("detectors", [])
        n_det = len(splitters) + 1
        while len(detectors) < n_det:
            detectors.append({"name": f"Channel {len(detectors) + 1}"})

        self._detector_widgets = []
        for i, det in enumerate(detectors[:n_det]):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(det.get("name", f"Channel {i + 1}"))
            lbl.setStyleSheet("font-weight: bold;")
            row.addWidget(lbl)
            row.addSpacing(6)
            row.addWidget(QtWidgets.QLabel("BP:"))
            bp_cb = _make_combo(self, self.probes, "has_trans")
            _set_combo(bp_cb, det.get("bandpass_probe_id"))
            bp_cb.currentIndexChanged.connect(self._schedule_recalc)
            row.addWidget(bp_cb, 1)
            row.addSpacing(4)
            row.addWidget(QtWidgets.QLabel("QE:"))
            qe_cb = _make_combo(self, self.probes, "has_qe")
            _set_combo(qe_cb, det.get("qe_probe_id"))
            qe_cb.currentIndexChanged.connect(self._schedule_recalc)
            row.addWidget(qe_cb, 1)
            container = QtWidgets.QWidget()
            container.setLayout(row)
            self.form_layout.addRow(f"Ch {i + 1}:", container)
            self._component_rows.append(container)
            self._detector_widgets.append({"bp": bp_cb, "qe": qe_cb, "name": det.get("name", f"Channel {i + 1}")})

        # Dye table
        self.dye_table = _DyeTableWidget(self.probes, db_path=self._db_path)
        dyes = cfg.get("dyes", {})
        if dyes:
            self.dye_table.set_selected_dyes(dyes)
        self.dye_table.dyeSelectionChanged.connect(self._schedule_recalc)
        self.form_layout.addRow("Dyes:", self.dye_table)
        self._component_rows.append(self.dye_table)

        # Parameters
        g_params = QtWidgets.QGroupBox("Parameters")
        pl = QtWidgets.QGridLayout(g_params)
        self.kappa2_spin = QtWidgets.QDoubleSpinBox()
        self.kappa2_spin.setRange(0, 4)
        self.kappa2_spin.setSingleStep(0.1)
        self.kappa2_spin.setValue(cfg.get("kappa2", 0.6667))
        self.n_spin = QtWidgets.QDoubleSpinBox()
        self.n_spin.setRange(1.0, 2.0)
        self.n_spin.setSingleStep(0.01)
        self.n_spin.setValue(cfg.get("n", 1.33))
        pl.addWidget(QtWidgets.QLabel("kappa²:"), 0, 0)
        pl.addWidget(self.kappa2_spin, 0, 1)
        pl.addWidget(QtWidgets.QLabel("n:"), 1, 0)
        pl.addWidget(self.n_spin, 1, 1)
        self.kappa2_spin.valueChanged.connect(self._schedule_recalc)
        self.n_spin.valueChanged.connect(self._schedule_recalc)
        self.form_layout.addRow(g_params)
        self._component_rows.append(g_params)

    # ── Signals ──

    def _connect_signals(self):
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self.btn_edit.clicked.connect(self._on_edit)
        self.btn_calc.clicked.connect(self.recalculate)
        self.btn_full.clicked.connect(self._on_open_full)
        self.auto_recalc_cb.toggled.connect(self._on_auto_recalc)

    # ── Preset management ──

    def _preset_path(self, name: str) -> Path:
        return OPTICAL_PRESETS_DIR / f"{name.strip()}.json"

    def _refresh_preset_list(self):
        self.preset_combo.blockSignals(True)
        current = self.preset_combo.currentText()
        self.preset_combo.clear()
        if OPTICAL_PRESETS_DIR.exists():
            for f in sorted(OPTICAL_PRESETS_DIR.glob("*.json")):
                self.preset_combo.addItem(f.stem)
        idx = self.preset_combo.findText(current)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, idx: int):
        name = self.preset_combo.currentText().strip()
        if not name:
            return
        path = self._preset_path(name)
        if not path.exists():
            return
        try:
            self._suppress_recalc = True
            cfg = load_easy_preset(str(path))
            # Detect graph-dict format (saved from Full Simulator) → convert
            if "nodes" in cfg:
                cfg = _graph_to_config(cfg)
            self._populate_form(cfg)
            self._suppress_recalc = False
            self._schedule_recalc()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Failed", str(exc))

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

    def _schedule_recalc(self):
        if self._suppress_recalc:
            return
        if self.auto_recalc_cb.isChecked():
            self._recalc_timer.start(300)

    # ── Config ──

    def _get_config(self) -> dict:
        lasers = self._lasers_edit.text() if self._lasers_edit else "488:1.0, 640:1.0"
        exci_pid = _combo_value(self._exci_combo)
        splitters = []
        for cb in self._splitter_widgets:
            spl_meta = cb.property("_splitter_type") or "Dichroic"
            splitters.append({"type": spl_meta, "probe_id": _combo_value(cb)})
        detectors = []
        for dw in self._detector_widgets:
            detectors.append({
                "name": dw["name"],
                "bandpass_probe_id": _combo_value(dw["bp"]),
                "qe_probe_id": _combo_value(dw["qe"]),
            })
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
            self.results_group.setVisible(True)
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
                self._suppress_recalc = True
                self._populate_form(cfg)
                self._suppress_recalc = False
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

    def _on_open_full(self):
        cfg = self._get_config()
        graph = build_easy_graph(cfg)
        from chisurf.plugins.core.lightpath_simulator.gui.tool import LightPathSimulatorWidget
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, LightPathSimulatorWidget):
            parent = parent.parentWidget()
        if parent is not None:
            parent.load_graph_from_dict(graph)
        else:
            w = LightPathSimulatorWidget()
            w.load_graph_from_dict(graph)
            w.show()

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
        self._populate_form(config)
        cached = config.get("_cached_results")
        if cached:
            self._fill_table(self.forster_table, cached.get("forster", {}))
            cm = cached.get("crosstalk_matrices", {})
            if cm:
                self._fill_table(self.ex_table, cm.get("excitation", {}))
                self._fill_table(self.em_table, cm.get("emission", {}))
                self._fill_table(self.det_table, cm.get("detected", {}))
                self.results_group.setVisible(True)
        self._suppress_recalc = False


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
