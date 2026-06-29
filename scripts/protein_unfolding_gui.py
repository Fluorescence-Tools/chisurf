# !chisurf: ipython
"""Protein unfolding FRET line — interactive GUI version.

Opens three fit windows in the ChiSurf MDI area so you can interact with them:
  • Gaussian model   — folded state
  • WLC model        — unfolded state
  • Lifetime mixer   — weighted combination of the two

The ``# !chisurf: ipython`` line above is a ChiSurf endpoint shebang.
It tells the Code Editor to always send this script to the IPython console
(``%run -i``), regardless of what the toolbar dropdown is set to.
Valid values: ``console`` · ``ipython`` · ``process``.

Usage
-----
Just press ▶ Run in the Code Editor — the shebang picks the right endpoint.

For headless/terminal use (no GUI needed) see: scripts/protein_unfolding_fret_line.py
"""
import sys
import numpy as np

# ── guard: must run inside ChiSurf (Console or IPython endpoint, not Process) ─
try:
    cs  # noqa: F821 — injected by Console/IPython endpoints
except NameError:
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Use the 'Console' or 'IPython' endpoint, not 'Process'.\n"
        "  Change the dropdown in the toolbar, then press ▶ again.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    sys.exit(0)

# ── parameters — edit these before running ────────────────────────────────────
TAU_D0     = 4.0    # donor lifetime without acceptor (ns)
R0         = 52.0   # Förster radius (Å)
GAUSS_MEAN = 35.0   # folded-state mean distance (Å)  →  E ≈ 0.91
GAUSS_SIG  = 4.0    # width (Å)
WLC_LC     = 80.0   # contour length (Å)  — kappa = Lp/Lc = 0.75 (valid regime)
WLC_LP     = 60.0   # persistence length (Å)

# ── create a synthetic TCSPC dataset and register it ─────────────────────────
print("Creating synthetic dataset ...")
t = np.linspace(0, 25, 4096)
y = np.random.default_rng(42).poisson(np.exp(-t / TAU_D0) * 1e4 + 1).astype(float)
dc = cs.core.data.DataCurve(x=t, y=y, name="protein_FRET_sim")
dc.experiment = cs.experiment.get("TCSPC")   # attach TCSPC experiment so models resolve
cs.imported_datasets.append(dc)
idx = len(cs.imported_datasets) - 1
print(f"  Registered as cs.imported_datasets[{idx}]")

# ── open three fit windows ────────────────────────────────────────────────────
print("Opening fit windows ...")
cs.macros.core_fit.add_fit(dataset_indices=[idx], model_name="FRET: FD (Gaussian)")
gauss_fit = cs.fits[-1]

cs.macros.core_fit.add_fit(dataset_indices=[idx], model_name="FRET: FD (Worm-like chain)")
wlc_fit = cs.fits[-1]

cs.macros.core_fit.add_fit(dataset_indices=[idx], model_name="Lifetime mixer")
mix_fit = cs.fits[-1]

# ── set Gaussian parameters ───────────────────────────────────────────────────
gm = gauss_fit.model
gm.fret_parameters.tauD0          = TAU_D0
gm.fret_parameters.forster_radius = R0
gm.fret_parameters.kappa2         = 2/3
gm.fret_parameters.xDOnly         = 0.0
gm.donor.lifetimes                = [TAU_D0]
gm.donor.amplitudes               = [1.0]
while len(gm.gaussians) > 0:
    gm.gaussians.pop()
gm.gaussians.append(mean=GAUSS_MEAN, sigma=GAUSS_SIG, x=1.0)

# ── set WLC parameters ────────────────────────────────────────────────────────
wm = wlc_fit.model
wm.fret_parameters.tauD0          = TAU_D0
wm.fret_parameters.forster_radius = R0
wm.fret_parameters.kappa2         = 2/3
wm.fret_parameters.xDOnly         = 0.0
wm.donor.lifetimes                = [TAU_D0]
wm.donor.amplitudes               = [1.0]
wm.chain_length                   = WLC_LC
wm.persistence_length             = WLC_LP

# ── wire the mixture: link Gaussian + WLC into the mixer ─────────────────────
mm = mix_fit.model
mm.append_model(gm, name="x_folded")
mm.append_model(wm, name="x_unfolded")
mm.fractions = [0.5, 0.5]   # start 50 % folded / 50 % unfolded

# ── print a quick FRET-line sweep ─────────────────────────────────────────────
print(f"\n{'f_unfold':>10}  {'E_FRET':>8}  {'<tau> ns':>10}")
print("-" * 34)
for f in np.linspace(0, 1, 11):
    mm.fractions = [1 - f, f]
    lt = mm.lifetime_spectrum
    amps, tauvec = lt[::2], lt[1::2]
    tau = float(np.dot(amps, tauvec) / amps.sum())
    print(f"{f:>10.2f}  {1 - tau/TAU_D0:>8.4f}  {tau:>10.4f}")

mm.fractions = [0.5, 0.5]

print("""
Three fit windows are open in the MDI area.
You can adjust any parameter interactively.

Console shortcuts (variables still in scope):
  gm   — Gaussian model   (folded)
  wm   — WLC model        (unfolded)
  mm   — Lifetime mixer

Examples:
  mm.fractions = [0.2, 0.8]            # 80% unfolded
  while len(gm.gaussians) > 0: gm.gaussians.pop()
  gm.gaussians.append(mean=40.0, sigma=4.0, x=1.0)
  wm.chain_length = 100.0
  wm.persistence_length = 50.0
""")
