"""Reusable table item that lazily renders a rich-HTML tooltip on hover.

:class:`TooltipItem` is domain-agnostic: it shows ``name`` in the cell and, the
first time the tooltip is requested, calls ``render_fn(key)`` to build the HTML
(e.g. a base64-encoded plot) and caches it. ``render_spectra_thumbnail`` is the
fluorescence-specific renderer used by the light-path simulator to draw
absorption / emission / transmission spectra; pair them as
``TooltipItem(name, key, render_fn=lambda k: render_spectra_thumbnail(k, adapter))``.
``SpectraTooltipItem`` is kept as a backwards-compatible alias.
"""
from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets


class TooltipItem(QtWidgets.QTableWidgetItem):
    """Table widget item that lazily renders an HTML tooltip on first hover.

    Parameters
    ----------
    name : str
        Display text for the cell.
    key : object
        Opaque key passed to *render_fn* (e.g. a probe id, row id, path).
    render_fn : callable or None
        ``callable(key) -> str`` returning an HTML snippet (typically a
        base64-encoded PNG). Called once and cached.
    """

    def __init__(self, name: str, key, render_fn=None):
        super().__init__(name)
        self._key = key
        self._render_fn = render_fn
        self._cached_html = None

    def data(self, role):
        """Return the cell data; lazily build and cache the tooltip HTML."""
        if role == QtCore.Qt.ToolTipRole:
            if self._cached_html is None and self._render_fn is not None:
                try:
                    html = self._render_fn(self._key) or ""
                except Exception:
                    html = ""
                self._cached_html = html
            return self._cached_html
        return super().data(role)


#: Backwards-compatible alias from when this item was spectra-specific.
SpectraTooltipItem = TooltipItem


def render_spectra_thumbnail(probe_id: int, adapter) -> str:
    """Render abs/em/transmission spectra as a small PNG in an HTML img tag.

    Parameters
    ----------
    probe_id : int
        Probe identifier in the MFDB.
    adapter : MFDatabaseAdapter or None
        Database adapter used to fetch spectra.

    Returns
    -------
    str
        HTML ``<img>`` tag with a base64-encoded PNG, or ``""`` when no
        spectra are available.
    """
    if adapter is None:
        return ""
    abs_spec = adapter.get_probe_spectrum(probe_id, "absorption")
    em_spec = adapter.get_probe_spectrum(probe_id, "emission")
    trans_spec = adapter.get_probe_spectrum(probe_id, "transmission")
    if not abs_spec and not em_spec and not trans_spec:
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
    if trans_spec:
        all_wl.extend(trans_spec[0])
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

    if trans_spec:
        pen = QtGui.QPen(QtGui.QColor("#44cc44"), 1.5)
        p.setPen(pen)
        wl, vals = trans_spec
        vmax = max(vals) if max(vals) > 0 else 1
        for i in range(len(wl) - 1):
            p.drawLine(
                int(to_px(wl[i])), int(h - mb - (vals[i] / vmax) * ph),
                int(to_px(wl[i + 1])), int(h - mb - (vals[i + 1] / vmax) * ph),
            )

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
