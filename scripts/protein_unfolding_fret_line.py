# !chisurf: process
"""Protein unfolding FRET line — headless / standalone.

Computes a two-state folding FRET line (Gaussian folded + WLC unfolded)
and writes results to text files. No GUI needed.

The ``# !chisurf: process`` shebang tells the Code Editor to run this as a
subprocess regardless of the toolbar dropdown. Valid values: console · ipython · process.

Run from terminal::

    conda activate arm64
    python scripts/protein_unfolding_fret_line.py

For the interactive GUI version see: scripts/protein_unfolding_gui.py
"""
import pathlib
import numpy as np
import chisurf
import chisurf.core.fitting.fit as fit_mod
from chisurf.core.data import DataCurve
from chisurf.core.models.tcspc.fret import GaussianModel, WormLikeChainModel
from chisurf.core.models.tcspc.lifetime import LifetimeMixtureModel

# ── parameters ────────────────────────────────────────────────────────────────
TAU_D0     = 4.0          # donor lifetime without acceptor (ns)
R0         = 52.0         # Förster radius (Å)
GAUSS_MEAN = 35.0         # folded-state mean distance (Å)  →  E ≈ 0.91
GAUSS_SIG  = 4.0          # width (Å)
WLC_LC     = 80.0         # contour length (Å)  — kappa = Lp/Lc = 0.75 (valid regime)
WLC_LP     = 60.0         # persistence length (Å)

FRACS = np.linspace(0, 1, 21)          # fraction unfolded: 0 → 1
TIME  = np.linspace(0, 25, 256)        # time axis for synthetic decays (ns)
OUT   = pathlib.Path(__file__).parent  # output folder = same dir as this script

# ── build Gaussian (folded) model ─────────────────────────────────────────────
print("Building Gaussian (folded) model ...")
dummy = DataCurve(x=TIME.copy(), y=np.ones(len(TIME)))
gauss_fit = fit_mod.Fit(model_class=GaussianModel, data=dummy)
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

# ── build WLC (unfolded) model ────────────────────────────────────────────────
print("Building WLC (unfolded) model ...")
wlc_fit = fit_mod.Fit(model_class=WormLikeChainModel, data=dummy)
wm = wlc_fit.model
wm.fret_parameters.tauD0          = TAU_D0
wm.fret_parameters.forster_radius = R0
wm.fret_parameters.kappa2         = 2/3
wm.fret_parameters.xDOnly         = 0.0
wm.donor.lifetimes                = [TAU_D0]
wm.donor.amplitudes               = [1.0]
wm.chain_length                   = WLC_LC
wm.persistence_length             = WLC_LP

# ── build mixture and sweep fraction unfolded ─────────────────────────────────
print("Sweeping fraction unfolded ...")
mix_fit = fit_mod.Fit(model_class=LifetimeMixtureModel, data=dummy)
mm = mix_fit.model
mm.append_model(gm, name="x_folded")
mm.append_model(wm, name="x_unfolded")

fracs, effs, taus, decays = [], [], [], []
for f in FRACS:
    mm.fractions = [1 - f, f]
    lt = mm.lifetime_spectrum              # interleaved [amp, tau, amp, tau, ...]
    amps, tauvec = lt[::2], lt[1::2]
    tau_avg = float(np.dot(amps, tauvec) / amps.sum())
    e = 1 - tau_avg / TAU_D0
    decay = sum(a * np.exp(-TIME / t) for a, t in zip(amps, tauvec) if t > 0)
    decay = decay / decay.sum() if decay.sum() > 0 else decay
    fracs.append(f); effs.append(e); taus.append(tau_avg); decays.append(decay)
    print(f"  f={f:.2f}  E={e:.4f}  <tau>={tau_avg:.4f} ns")

# ── WLC parameter sweep (grid of Lc × Lp) ────────────────────────────────────
print("\nWLC parameter sweep ...")
sweep_rows = []
for lc in [60, 80, 100, 120]:
    for lp in [40, 60, 80, 100]:
        print(f"  Lc={lc} Å  Lp={lp} Å ...")
        w2 = fit_mod.Fit(model_class=WormLikeChainModel, data=dummy).model
        w2.fret_parameters.tauD0          = TAU_D0
        w2.fret_parameters.forster_radius = R0
        w2.fret_parameters.kappa2         = 2/3
        w2.fret_parameters.xDOnly         = 0.0
        w2.donor.lifetimes                = [TAU_D0]
        w2.donor.amplitudes               = [1.0]
        w2.chain_length                   = lc
        w2.persistence_length             = lp

        m2 = fit_mod.Fit(model_class=LifetimeMixtureModel, data=dummy).model
        m2.append_model(gm, name="x_folded")
        m2.append_model(w2, name="x_unfolded")
        for f in FRACS:
            m2.fractions = [1 - f, f]
            lt = m2.lifetime_spectrum
            amps, tauvec = lt[::2], lt[1::2]
            tau_avg = float(np.dot(amps, tauvec) / amps.sum())
            sweep_rows.append([lp, lc, f, 1 - tau_avg / TAU_D0, tau_avg])

# ── write output files ────────────────────────────────────────────────────────
print("\nWriting output files ...")

np.savetxt(
    OUT / "unfolding_fret_line.txt",
    np.column_stack([fracs, effs, taus]),
    header=(
        f"tau_D0={TAU_D0} ns  R0={R0} A  "
        f"Gaussian R_mean={GAUSS_MEAN} A sigma={GAUSS_SIG} A  "
        f"WLC Lc={WLC_LC} A Lp={WLC_LP} A\n"
        "fraction_unfolded\tE_FRET\ttau_avg_ns"
    ),
    delimiter="\t", fmt="%.6f",
)

np.savetxt(
    OUT / "unfolding_decays.txt",
    np.column_stack([TIME] + decays),
    header="time_ns\t" + "\t".join(f"f={f:.2f}" for f in fracs),
    delimiter="\t", fmt="%.8f", comments="",
)

np.savetxt(
    OUT / "wlc_sweep_fret_lines.txt",
    np.array(sweep_rows),
    header="Lp_A\tLc_A\tfraction_unfolded\tE_FRET\ttau_avg_ns",
    delimiter="\t", fmt=["%.1f", "%.1f", "%.6f", "%.6f", "%.6f"], comments="",
)

print(f"Done. Files written to {OUT}/")
