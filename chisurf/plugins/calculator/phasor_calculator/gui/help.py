"""Help content + modal dialog for the phasor calculator.

The prose is a compact, reference-free summary of the phasor approach, condensed
from the PhasorPy documentation and tutorials (no citations reproduced). Kept in
its own module so the tool widget stays lean and the text is easy to edit.
"""

from __future__ import annotations

from qtpy import QtCore, QtWidgets

HELP_HTML = """
<h2>The phasor plot</h2>
<p>The <b>phasor approach</b> is a fit-free way to look at time-resolved (and
spectral) fluorescence. Each measurement is transformed into a single point with
coordinates <b>g</b> (horizontal) and <b>s</b> (vertical) at a chosen modulation
frequency. Because it needs no exponential fitting, the analysis is fast,
low-dimensional, and intuitive: similar signals cluster together, and every point
maps back to the pixel it came from.</p>

<h3>Universal semicircle</h3>
<p>The white half-circle (centre <i>(0.5, 0)</i>, radius <i>0.5</i>) is the
<b>universal circle</b>. Every <b>single-exponential</b> lifetime lies exactly on
it: short lifetimes sit near <i>(1, 0)</i>, long lifetimes near the origin. Any
point <i>inside</i> the semicircle is a <b>mixture</b> of several lifetime
components.</p>

<h3>Apparent lifetimes</h3>
<p>From a point's phase angle and its distance from the origin (modulation) two
apparent lifetimes are read off: the <b>phase lifetime</b> &tau;<sub>&phi;</sub>
and the <b>modulation lifetime</b> &tau;<sub>M</sub>. For a true
single-exponential they are equal; for a mixture &tau;<sub>&phi;</sub> &lt;
&tau;<sub>M</sub>.</p>

<h3>Linear combinations &amp; fractions</h3>
<p>Phasors add linearly. A pixel containing several species falls inside the
polygon whose vertices are the pure-component phasors, and its distance to each
vertex is inversely proportional to that component's fractional intensity — the
closer the point, the larger the contribution. This lets fractions of multiple
species be recovered directly, without fitting.</p>

<h3>FRET trajectory</h3>
<p>As a donor is progressively quenched by FRET its lifetime shortens, tracing a
curve from the donor-only position toward <i>(1, 0)</i>. The trajectory shows
where donor phasors should fall for increasing FRET efficiency.</p>

<h3>Overlays in this tool</h3>
<ul>
<li><b>Iso-lifetime grid</b> — lines of constant phase and modulation lifetime.</li>
<li><b>Lifetime ticks</b> — reference lifetimes marked on the semicircle.</li>
<li><b>Polar grid</b> — concentric circles and angular spokes for reading phase
and modulation.</li>
<li><b>FRET trajectory</b> — the quenched-donor curve for the donor lifetime you
set.</li>
<li><b>Two-component line</b> — the mixing line between two component phasors.</li>
<li><b>Mixing region</b> — the two-component mixing geometry plus the
fraction-weighted mixture point (set the fraction of component 1).</li>
<li><b>Cursor</b> — a circular gating-cursor outline at a chosen g, s position.</li>
</ul>
"""


class PhasorHelpDialog(QtWidgets.QDialog):
    """Modal, scrollable help window describing the phasor plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phasor plot — help")
        self.setModal(True)
        self.resize(560, 620)
        layout = QtWidgets.QVBoxLayout(self)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(HELP_HTML)
        layout.addWidget(browser, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(self.accept)
        layout.addWidget(buttons, 0, QtCore.Qt.AlignRight)
