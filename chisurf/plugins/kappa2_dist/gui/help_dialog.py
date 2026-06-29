"""Modal help dialog for the k² Distribution Calculator."""

from __future__ import annotations

from qtpy import QtCore, QtWidgets


class Kappa2DistHelpDialog(QtWidgets.QDialog):
    """Modal help dialog with theory explanation for the k² orientation factor."""

    TITLE = "About the κ² orientation factor distribution"
    WIDTH = 560
    HEIGHT = 500

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.resize(self.WIDTH, self.HEIGHT)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        text = QtWidgets.QTextEdit(self)
        text.setReadOnly(True)
        text.setHtml(self._content())

        layout.addWidget(text)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    @staticmethod
    def _content() -> str:
        return """
<h2>The κ² Orientation Factor</h2>

<p>
  In FRET, the energy transfer rate depends not only on the donor-acceptor
  distance but also on the relative orientation of the transition dipoles.
  This geometric dependence is captured by the orientation factor κ².
</p>
<p>
  κ² ranges from 0 (perpendicular dipoles, no transfer) to 4 (perfectly
  aligned head-to-tail dipoles, maximum transfer).  When both dyes rotate
  freely on a time scale much faster than the donor lifetime, the
  orientation factor averages to the well-known value of 2/3.
</p>
<p>
  In practice dyes are often partially restricted by their linker
  geometry and local environment.  This calculator estimates the full
  distribution p(κ²) under different motional models, so that the
  effect of orientational heterogeneity on the apparent FRET distance
  can be assessed.
</p>

<h3>Wobbling-in-Cone (WIC) model</h3>
<p>
  Each dye is modelled as confined to wobble inside a cone.  The cone
  half-angle is derived from the order parameter S², which is obtained
  from the residual anisotropy of the dye.  S² = 0 corresponds to
  completely free rotation (isotropic), while S² = 1 means the dye is
  rigidly immobilised.
</p>
<p>
  The distribution is computed by sampling over all possible dipole
  orientations within the two cones.  The angle δ between the cone
  symmetry axes can either be fixed (using the FRET-sensitised acceptor
  residual anisotropy) or left free.
</p>

<h3>Diffusion-with-Traps (DWT) model</h3>
<p>
  A fraction S² of each dye population is treated as statically trapped
  (fixed orientation), while the remaining fraction 1−S² is freely
  diffusing.  Four sub-populations arise: both free, donor trapped and
  acceptor free, donor free and acceptor trapped, and both trapped.
  Each sub-population has its own effective orientation factor.
</p>
<p>
  The FRET efficiency enters the calculation because it determines the
  effective averaging time.  Higher efficiency means a shorter donor
  lifetime, which weights the trapped (anisotropic) fraction more
  heavily relative to the free (isotropic) fraction.
</p>

<h3>Isotropic model</h3>
<p>
  Both dyes rotate freely and isotropically.  This is the classical
  limiting case where the distribution p(κ²) has a known analytic form,
  peaked near zero and falling off towards κ² = 4.  The distribution
  depends on the specific geometry of the dipole pair for each
  configuration, but the ensemble average always gives κ² = 2/3.
</p>

<h3>Output fields</h3>
<table>
  <tr><td><strong>Mean κ²</strong></td>
      <td>Weighted average of the computed distribution.</td></tr>
  <tr><td><strong>SD κ²</strong></td>
      <td>Standard deviation of the distribution.</td></tr>
  <tr><td><strong>Rapp / R_DA</strong></td>
      <td>Correction factor for the apparent FRET distance, averaged
          over the κ² distribution.</td></tr>
  <tr><td><strong>δ (deg)</strong></td>
      <td>Angle between the cone symmetry axes (WIC model, when
          r_AD known is checked).</td></tr>
</table>

<h3>References</h3>
<ul>
  <li>Sindbert, S. et al. (2011). <em>J. Am. Chem. Soc.</em>,
      133(8), 2463–2480.</li>
  <li>Peulen, T. O. et al. (2017). <em>J. Phys. Chem. B</em>,
      121(35), 8211–8241.</li>
</ul>
"""
