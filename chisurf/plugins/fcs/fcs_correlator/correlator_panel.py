"""Correlator settings model and custom AutoForm sections."""

from __future__ import annotations

import pathlib
import typing

import numpy as np
import tttrlib
from qtpy import QtCore, QtWidgets

import chisurf as cs
from chisurf.core.dataspec import load_view_spec
from chisurf.core.fluorescence.fcs.channel_setups import load_fcs_channel_setups
from chisurf.gui.autoform import register_section

_GUI_DIR = pathlib.Path(__file__).resolve().parent


def parse_microtime_ranges(
    s: str,
) -> typing.List[typing.Tuple[int, int]] | None:
    if not s:
        return None
    text = str(s).strip()
    if not text:
        return None
    segments = []
    for item in text.replace(",", ";").split(";"):
        item = item.strip()
        if item:
            segments.append(item)
    if not segments:
        return None
    ranges: typing.List[typing.Tuple[int, int]] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if ":" in seg:
            a_txt, b_txt = seg.split(":", 1)
        else:
            pos = seg.rfind("-")
            if pos <= 0:
                a_txt = seg
                b_txt = seg
            else:
                a_txt = seg[:pos]
                b_txt = seg[pos + 1:]
        a = int(a_txt.strip())
        b = int(b_txt.strip())
        if a <= b:
            ranges.append((a, b))
        else:
            ranges.append((b, a))
    return ranges if ranges else None


def parse_channels(s: str) -> typing.List[int]:
    if not s:
        return []
    return [int(x) for x in s.replace(",", " ").split()]


class CorrelatorSettingsModel:
    def __init__(self):
        self.n_bins = int(
            cs.core.settings.cs_settings.get("correlator", {}).get("B", 3)
        )
        self.n_casc = int(
            cs.core.settings.cs_settings.get("correlator", {}).get(
                "number_of_cascades", 20
            )
        )
        self.n_splits = int(
            cs.core.settings.cs_settings.get("correlator", {}).get("split", 1)
        )
        self.make_fine = bool(
            cs.core.settings.cs_settings.get("correlator", {}).get("fine", False)
        )
        self.microtime_binning = int(
            cs.core.settings.cs_settings.get("correlator", {}).get(
                "microtime_binning", 1
            )
        )
        self.channel_a = ""
        self.channel_b = ""
        self.microtime_range_a = ""
        self.microtime_range_b = ""

        self._tttr: tttrlib.TTTR | None = None
        self._correlations: typing.List[dict] = []
        self._channel_defs: dict = {}
        self._analysis_folder = pathlib.Path()
        self._output_subdir = pathlib.Path("cr5")

        self._fcs_presets: typing.List[dict] = []
        self._fcs_preset_detectors: dict = {}
        self._fcs_preset_corr: dict = {}

        self._form: typing.Any = None

    def view_spec(self):
        return load_view_spec(_GUI_DIR / "correlator.view.json")

    def get_correlation_settings(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "n_casc": self.n_casc,
            "make_fine": self.make_fine,
        }

    def correlation_series(self):
        import pyqtgraph as pg

        n = len(self._correlations)
        return [
            {
                "x": c["x"],
                "y": c["y"],
                "name": f"chunk {i}",
                "color": pg.intColor(i, hues=max(n, 6)),
            }
            for i, c in enumerate(self._correlations)
        ]

    def correlate_data(self, parent_widget: QtWidgets.QWidget | None = None) -> None:
        if self._tttr is None or len(self._tttr) == 0:
            QtWidgets.QMessageBox.warning(
                parent_widget,
                "No Photons Selected",
                "No photons selected for correlation. Please load data.",
            )
            return

        n_chunks = self.n_splits
        ch1 = parse_channels(self.channel_a)
        ch2 = parse_channels(self.channel_b)
        if not ch1 or not ch2:
            used = list(map(int, self._tttr.get_used_routing_channels()))
            if not ch1:
                ch1 = used
            if not ch2:
                ch2 = used

        settings = self.get_correlation_settings()
        self._correlations.clear()

        progress = QtWidgets.QProgressDialog(
            "Computing correlations...",
            "Cancel",
            0,
            n_chunks,
            parent_widget,
        )
        progress.setWindowTitle("Correlation Progress")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        for i, chunk in enumerate(self._split_array(self._tttr, n_chunks)):
            if progress.wasCanceled():
                break
            if chunk is None or len(chunk.macro_times) == 0:
                continue
            result = self._correlate_one(chunk, ch1, ch2, settings, i)
            if result is not None:
                self._correlations.append(result)
            progress.setValue(i + 1)
            progress.raise_()
            QtWidgets.QApplication.processEvents()

        progress.close()

        if self._form is not None:
            self._form.refresh_plots()

    def _split_array(self, tttr, n):
        chunk_size = max(1, len(tttr) // n)
        return [tttr[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]

    def _correlate_one(self, tttr, ch1, ch2, settings, idx):
        t = tttr.macro_times
        mask_a = tttrlib.TTTRMask()
        mask_b = tttrlib.TTTRMask()
        mask_a.select_channels(tttr, ch1, mask=True)
        mask_b.select_channels(tttr, ch2, mask=True)
        m_a = mask_a.mask.astype(bool)
        m_b = mask_b.mask.astype(bool)

        mta = parse_microtime_ranges(self.microtime_range_a)
        mtb = parse_microtime_ranges(self.microtime_range_b)
        if mta:
            mm_a = tttrlib.TTTRMask()
            mm_a.select_microtime_ranges(tttr, mta)
            mm_a.flip()
            m_a = np.logical_and(m_a, mm_a.mask.astype(bool))
        if mtb:
            mm_b = tttrlib.TTTRMask()
            mm_b.select_microtime_ranges(tttr, mtb)
            mm_b.flip()
            m_b = np.logical_and(m_b, mm_b.mask.astype(bool))

        w1 = np.array(m_a, dtype=np.float64)
        w2 = np.array(m_b, dtype=np.float64)
        sw1 = w1.sum()
        sw2 = w2.sum()
        if sw1 <= 0.0 or sw2 <= 0.0:
            return None

        dT = tttr.header.macro_time_resolution * 1000.0
        if len(t) > 100:
            t_start = np.percentile(t, 0.1)
            t_end = np.percentile(t, 99.9)
        else:
            t_start = t[0] if len(t) > 0 else 0.0
            t_end = t[-1] if len(t) > 0 else 0.0
        dur = (t_end - t_start) * dT

        correlator = tttrlib.Correlator(**settings)
        # Symmetric (Schätzel) normalization — normalizes each lag by the count
        # rate in the overlapping sub-intervals instead of the global mean count
        # rate, removing the long-lag upturn artifact near the chunk duration.
        try:
            correlator.method = "laurence"
        except Exception:
            pass
        correlator.set_macrotimes(t, t)
        correlator.set_weights(w1, w2)
        if self.make_fine:
            b = self.microtime_binning
            n_mt = tttr.get_number_of_micro_time_channels()
            mt = tttr.micro_times
            if b > 1:
                mt = mt // b
                n_mt = (n_mt + b - 1) // b
            correlator.set_microtimes(mt, mt, n_mt)
            dt = tttr.header.micro_time_resolution * b * 1000.0
        else:
            dt = dT
        x = correlator.x_axis * dt
        y = np.asarray(correlator.correlation, dtype=float)
        # Multi-tau produces lag times set by the cascade count, which can run
        # past the chunk's actual measured duration. Correlation values at lags
        # beyond the duration are meaningless (ever fewer photon pairs). Keep the
        # lag grid intact (so chunks stay averageable) but flatten those points to
        # the uncorrelated baseline G=1 (zero correlation amplitude). ``dur`` and
        # ``x`` are both in milliseconds.
        if dur > 0.0:
            y[x > dur] = 1.0
        return {
            "x": x.tolist(),
            "y": y.tolist(),
            "correlation_settings": settings,
            "chunk": idx,
            "duration": dur / 1000.0,
            "channel_a": {
                "channels": ch1,
                "microtime_range": mta,
                "counts": sw1,
            },
            "channel_b": {
                "channels": ch2,
                "microtime_range": mtb,
                "counts": sw2,
            },
        }

    def load_fcs_presets(self, setup_name, detectors) -> None:
        cfg = load_fcs_channel_setups()
        setups = cfg.get("setups", {}) if isinstance(cfg, dict) else {}
        block = setups.get(setup_name or "", {}) if isinstance(setups, dict) else {}
        pairs = block.get("pairs", []) if isinstance(block, dict) else []
        if not isinstance(pairs, list):
            pairs = []
        if not pairs and detectors:
            # No FCS preset block for this setup: derive sensible defaults from
            # the detectors — an ACF per detector plus a CCF for each pair.
            pairs = self._default_pairs_from_detectors(detectors)
        self._fcs_presets = pairs
        self._fcs_preset_detectors = detectors or {}
        self._fcs_preset_corr = block.get("correlator", {}) if isinstance(block, dict) else {}

    @staticmethod
    def _default_pairs_from_detectors(detectors: dict) -> typing.List[dict]:
        names = [n for n in detectors if isinstance(n, str) and n.strip()]
        pairs: typing.List[dict] = []
        for n in names:
            pairs.append({"name": f"{n} ACF", "channel_a": n, "channel_b": n, "kind": "ACF"})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pairs.append(
                    {"name": f"{a}×{b} CCF", "channel_a": a, "channel_b": b, "kind": "CCF"}
                )
        return pairs

    def apply_preset(self, index: int) -> bool:
        if index <= 0 or not self._fcs_presets:
            return False
        try:
            pair = self._fcs_presets[index - 1]
        except Exception:
            return False
        dets = self._fcs_preset_detectors or {}
        try:
            cha_name = str(pair.get("channel_a", ""))
            chb_name = str(pair.get("channel_b", ""))
        except Exception:
            return False
        da = dets.get(cha_name, {}) if isinstance(dets, dict) else {}
        db = dets.get(chb_name, {}) if isinstance(dets, dict) else {}
        chs_a = da.get("chs", []) or []
        chs_b = db.get("chs", []) or chs_a
        if chs_a:
            self.channel_a = ",".join(map(str, chs_a))
        if chs_b:
            self.channel_b = ",".join(map(str, chs_b))
        mta = da.get("micro_time_ranges", []) or []
        mtb = db.get("micro_time_ranges", []) or []
        if mta:
            self.microtime_range_a = ";".join(f"{a}-{b}" for a, b in mta)
        if mtb:
            self.microtime_range_b = ";".join(f"{a}-{b}" for a, b in mtb)
        corr = dict(self._fcs_preset_corr)
        pc = pair.get("correlator")
        if isinstance(pc, dict):
            corr.update(pc)
        if "n_bins" in corr:
            self.n_bins = int(corr["n_bins"])
        if "n_casc" in corr:
            self.n_casc = int(corr["n_casc"])
        if "make_fine" in corr:
            self.make_fine = bool(corr["make_fine"])
        if "microtime_binning" in corr:
            self.microtime_binning = int(corr["microtime_binning"])
        return True


# ---- Custom AutoForm sections ----------------------------------------------


@register_section("fcs_presets")
class _FcsPresetCombo(QtWidgets.QWidget):
    AUTOFORM_REFRESH = True

    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QtWidgets.QLabel("FCS Preset:")
        layout.addWidget(lbl)
        self.combo = QtWidgets.QComboBox()
        self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo.addItem("")
        self.combo.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self.combo, 1)

    def _on_changed(self, index: int) -> None:
        self._model.apply_preset(index)

    def refresh(self) -> None:
        self.combo.blockSignals(True)
        current = self.combo.currentText()
        self.combo.clear()
        self.combo.addItem("")
        for p in self._model._fcs_presets:
            try:
                cha = str(p.get("channel_a", ""))
                chb = str(p.get("channel_b", ""))
                nm = str(p.get("name", ""))
            except Exception:
                continue
            if not nm:
                nm = f"{cha}x{chb}" if cha != chb else f"{cha}_ACF" if cha else "(unnamed)"
            self.combo.addItem(nm)
        idx = self.combo.findText(current)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)


@register_section("channel_combos")
class _ChannelComboWidget(QtWidgets.QWidget):
    AUTOFORM_REFRESH = True

    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl_a = QtWidgets.QLabel("A:")
        layout.addWidget(lbl_a)
        self.combo_a = QtWidgets.QComboBox()
        self.combo_a.currentIndexChanged.connect(lambda i: self._on_combo("a"))
        layout.addWidget(self.combo_a, 1)

        lbl_b = QtWidgets.QLabel("B:")
        layout.addWidget(lbl_b)
        self.combo_b = QtWidgets.QComboBox()
        self.combo_b.currentIndexChanged.connect(lambda i: self._on_combo("b"))
        layout.addWidget(self.combo_b, 1)

    def _on_combo(self, side: str) -> None:
        combo = self.combo_a if side == "a" else self.combo_b
        ch_edit = "channel_a" if side == "a" else "channel_b"
        mt_edit = "microtime_range_a" if side == "a" else "microtime_range_b"
        key = combo.currentText()
        if not key:
            return
        entries = self._model._channel_defs.get(key)
        if not entries:
            return
        all_chs = []
        for e in entries:
            chs = e.get("detector_chs", [])
            if isinstance(chs, (list, tuple)):
                all_chs.extend(chs)
        seen = set()
        uniq = []
        for c in all_chs:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        setattr(self._model, ch_edit, ",".join(str(c) for c in uniq))
        segs = []
        for e in entries:
            r = e.get("micro_time_range")
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                segs.append(f"{int(r[0])}-{int(r[1])}")
        setattr(self._model, mt_edit, ";".join(segs))

    def refresh(self) -> None:
        keys = list(self._model._channel_defs.keys())
        try:
            keys.sort()
        except Exception:
            pass
        for combo in (self.combo_a, self.combo_b):
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItems(keys)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)


@register_section("correlate_controls")
class _CorrelateControls(QtWidgets.QWidget):
    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn = QtWidgets.QPushButton("Correlate")
        self.btn.setStyleSheet(
            "QPushButton { background-color: #1f7a1f; color: white; "
            "border: 1px solid #166016; border-radius: 4px; "
            "padding: 4px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #249124; }"
        )
        self.btn.clicked.connect(self._on_correlate)
        layout.addWidget(self.btn)

        self._status = QtWidgets.QLabel("No data loaded.")
        layout.addWidget(self._status, 1)

    def _on_correlate(self) -> None:
        parent = self.window() if self.window() else self
        self._model.correlate_data(parent_widget=parent)
        n = len(self._model._correlations)
        self._status.setText(f"{n} chunk(s) correlated." if n else "Correlation empty.")

    def update_status(self, text: str) -> None:
        self._status.setText(text)
