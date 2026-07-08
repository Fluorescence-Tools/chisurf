"""A/B correctness tests: numba H2MM port vs. the reference ``H2MM_C`` library.

The port in :mod:`chisurf.plugins.burst.burst_h2mm.core.h2mm` is asserted to be
numerically equivalent to the reference C/Cython implementation
(``H2MM_C`` / ``h2mm_c`` on PyPI) on identical data.  Photon data is simulated
and carried through a real ``tttrlib.TTTR`` object and the plugin's own
extraction path, so the whole A-side pipeline is exercised.

These tests are skipped when ``H2MM_C`` is not installed (it is an optional,
test-only comparison dependency, not a runtime dependency of ChiSurf).
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pandas as pd
import pytest

from chisurf.plugins.burst.burst_h2mm.core import h2mm as H
from chisurf.plugins.burst.burst_h2mm.core.photons import (
    StreamDef,
    bursts_from_dataframe,
)

h2mm_c = pytest.importorskip("H2MM_C")
tttrlib = pytest.importorskip("tttrlib")

# ---------------------------------------------------------------------------
# Shared simulation → tttrlib.TTTR → plugin extraction + reference lists
# ---------------------------------------------------------------------------


def _simulate_via_tttrlib(gt, n_bursts, burst_len, rate=0.25, seed=42):
    """Simulate an HMM photon stream, pack it into a real ``tttrlib.TTTR``.

    Returns the plugin-side :class:`BurstPhotons` (A) and the reference-side
    per-burst ``(indexes, times)`` lists (B), built from the *same* TTTR.
    """
    rng = np.random.default_rng(seed)
    times_local = [
        np.concatenate([[0], np.cumsum(rng.poisson(1.0 / rate, burst_len - 1) + 1)]).astype(np.int64)
        for _ in range(n_bursts)
    ]
    streams_local = H.simulate_bursts(gt, times_local, seed=seed + 7)

    macro, chan, rows = [], [], []
    off, base = 0, 0
    for t, s in zip(times_local, streams_local):
        macro.append((t + base).astype(np.uint64))
        chan.append(s.astype(np.int8))
        rows.append(("sim.spc", off, off + len(t)))
        off += len(t)
        base += int(t[-1]) + 100000  # large inter-burst gap keeps bursts distinct

    macro = np.concatenate(macro).astype(np.uint64)
    chan = np.concatenate(chan).astype(np.int8)
    micro = np.zeros(macro.size, dtype=np.uint16)
    et = np.zeros(macro.size, dtype=np.int8)

    tttr = tttrlib.TTTR()
    tttr.append_events(macro, micro, chan, et, False, 0)
    assert np.array_equal(np.asarray(tttr.macro_times), macro)

    df = pd.DataFrame(rows, columns=["First File", "First Photon", "Last Photon"])
    n_streams = gt.n_streams
    stream_defs = [StreamDef(f"s{i}", [i]) for i in range(n_streams)]
    data = bursts_from_dataframe(df, {"sim.spc": tttr}, stream_defs, min_photons=1)

    mt_all = np.asarray(tttr.macro_times)
    ch_all = np.asarray(tttr.routing_channels)
    ref_idx, ref_times = [], []
    for _, fp, lp in rows:
        ref_idx.append(ch_all[fp:lp].astype(np.uint32))
        ref_times.append((mt_all[fp:lp] - mt_all[fp]).astype(np.uint64))
    return data, ref_idx, ref_times


def _align(prior, trans, obs):
    """Return the model matrices with states sorted by stream-0 emission."""
    order = np.argsort(-obs[:, 0])
    return prior[order], trans[np.ix_(order, order)], obs[order]


def _two_state_gt():
    return H.H2mmModel(
        np.array([0.5, 0.5]),
        np.array([[0.990, 0.010], [0.020, 0.980]]),
        np.array([[0.80, 0.20], [0.25, 0.75]]),
    )


def _three_state_gt():
    return H.H2mmModel(
        np.array([1 / 3, 1 / 3, 1 / 3]),
        np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]),
        np.array([[0.85, 0.15], [0.50, 0.50], [0.15, 0.85]]),
    )


# ---------------------------------------------------------------------------
# A/B-1 — forward algorithm: log-likelihood of a fixed model must match
# ---------------------------------------------------------------------------


def test_forward_loglik_matches_reference():
    gt = _two_state_gt()
    data, ref_idx, ref_times = _simulate_via_tttrlib(gt, n_bursts=250, burst_len=70, seed=1)

    prior = np.array([0.5, 0.5])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    obs = np.array([[0.7, 0.3], [0.3, 0.7]])

    ref_ll = h2mm_c.H2MM_arr([h2mm_c.h2mm_model(prior.copy(), trans.copy(), obs.copy())], ref_idx, ref_times)[0].loglik
    mine_ll = H.optimize(H.H2mmModel(prior.copy(), trans.copy(), obs.copy()), data, max_iter=1, tol=0.0).loglik

    assert mine_ll == pytest.approx(ref_ll, rel=1e-6, abs=1e-4)


# ---------------------------------------------------------------------------
# A/B-2 — one Baum-Welch step: M-step (ρ tensor / ξ) must match
# ---------------------------------------------------------------------------


def test_one_em_step_matches_reference():
    gt = _two_state_gt()
    data, ref_idx, ref_times = _simulate_via_tttrlib(gt, n_bursts=250, burst_len=70, seed=2)

    prior = np.array([0.5, 0.5])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    obs = np.array([[0.7, 0.3], [0.3, 0.7]])

    with contextlib.redirect_stdout(io.StringIO()):
        ref = h2mm_c.EM_H2MM_C(h2mm_c.h2mm_model(prior.copy(), trans.copy(), obs.copy()), ref_idx, ref_times, max_iter=1)
    mine = H.optimize(H.H2mmModel(prior.copy(), trans.copy(), obs.copy()), data, max_iter=1, tol=0.0)

    rp, rt, ro = _align(ref.prior, ref.trans, ref.obs)
    mp, mt, mo = _align(mine.prior, mine.trans, mine.obs)
    assert np.abs(rp - mp).max() < 1e-5
    assert np.abs(rt - mt).max() < 1e-5
    assert np.abs(ro - mo).max() < 1e-5


# ---------------------------------------------------------------------------
# A/B-3 — full convergence from an identical start (2- and 3-state)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gt, start",
    [
        (
            _two_state_gt(),
            (
                np.array([0.5, 0.5]),
                np.array([[0.95, 0.05], [0.05, 0.95]]),
                np.array([[0.7, 0.3], [0.3, 0.7]]),
            ),
        ),
        (
            _three_state_gt(),
            (
                np.array([1 / 3, 1 / 3, 1 / 3]),
                np.array([[0.94, 0.03, 0.03], [0.03, 0.94, 0.03], [0.03, 0.03, 0.94]]),
                np.array([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]]),
            ),
        ),
    ],
    ids=["2-state", "3-state"],
)
def test_converged_model_matches_reference(gt, start):
    n = gt.n_states
    data, ref_idx, ref_times = _simulate_via_tttrlib(
        gt, n_bursts=350, burst_len=90, seed=10 + n
    )
    prior, trans, obs = start

    with contextlib.redirect_stdout(io.StringIO()):
        ref = h2mm_c.EM_H2MM_C(
            h2mm_c.h2mm_model(prior.copy(), trans.copy(), obs.copy()),
            ref_idx, ref_times, max_iter=500,
        )
    mine = H.optimize(
        H.H2mmModel(prior.copy(), trans.copy(), obs.copy()), data, max_iter=500, tol=1e-10
    )

    # Converged log-likelihood and BIC agree.
    assert mine.loglik == pytest.approx(ref.loglik, rel=1e-5, abs=1e-2)
    assert mine.bic == pytest.approx(ref.bic, rel=1e-5, abs=1e-2)

    # Converged model parameters agree (after label alignment).
    rp, rt, ro = _align(ref.prior, ref.trans, ref.obs)
    mp, mt, mo = _align(mine.prior, mine.trans, mine.obs)
    assert np.abs(rp - mp).max() < 5e-3
    assert np.abs(rt - mt).max() < 5e-3
    assert np.abs(ro - mo).max() < 5e-3


# ---------------------------------------------------------------------------
# A/B-4 — Viterbi most-likely path agreement
# ---------------------------------------------------------------------------


def test_viterbi_path_matches_reference():
    gt = _two_state_gt()
    data, ref_idx, ref_times = _simulate_via_tttrlib(gt, n_bursts=300, burst_len=80, seed=5)

    prior = np.array([0.5, 0.5])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    obs = np.array([[0.7, 0.3], [0.3, 0.7]])

    with contextlib.redirect_stdout(io.StringIO()):
        ref = h2mm_c.EM_H2MM_C(h2mm_c.h2mm_model(prior.copy(), trans.copy(), obs.copy()), ref_idx, ref_times, max_iter=500)
        ref_vit = h2mm_c.viterbi_path(ref, ref_idx, ref_times)
    mine = H.optimize(H.H2mmModel(prior.copy(), trans.copy(), obs.copy()), data, max_iter=500, tol=1e-10)

    ref_path = np.concatenate(list(ref_vit[0]))
    mine_path, _ = H.viterbi(mine, data)

    # Map each model's state labels to a canonical order before comparing.
    ref_relabel = {int(s): i for i, s in enumerate(np.argsort(-ref.obs[:, 0]))}
    mine_relabel = {int(s): i for i, s in enumerate(np.argsort(-mine.obs[:, 0]))}
    ref_norm = np.array([ref_relabel[int(s)] for s in ref_path])
    mine_norm = np.array([mine_relabel[int(s)] for s in mine_path])

    agreement = float(np.mean(ref_norm == mine_norm))
    assert agreement > 0.999, f"Viterbi agreement only {agreement:.4f}"


# ---------------------------------------------------------------------------
# A/B-5 — wall-clock benchmark: numba port vs. reference H2MM_C
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("n_states", [2, 3], ids=["2-state", "3-state"])
def test_benchmark_vs_reference(n_states, capsys):
    """Time one EM run against ``H2MM_C`` on identical data (perf A/B guard).

    Companion to the correctness tests above: the numba engine must stay
    competitive with the reference C implementation.  Timing is reported as
    ``ms / EM-iteration`` so the comparison is independent of how many
    iterations each engine happens to run, and the numba JIT is warmed up
    first so compilation is never timed.  Marked ``slow`` (excluded from
    default runs); run explicitly with ``-m slow -s`` to see the numbers.
    """
    gt = _three_state_gt() if n_states == 3 else _two_state_gt()
    data, ref_idx, ref_times = _simulate_via_tttrlib(
        gt, n_bursts=1500, burst_len=100, seed=100 + n_states
    )

    if n_states == 2:
        prior = np.array([0.5, 0.5])
        trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        obs = np.array([[0.7, 0.3], [0.3, 0.7]])
    else:
        prior = np.full(3, 1 / 3)
        trans = np.array([[0.94, 0.03, 0.03], [0.03, 0.94, 0.03], [0.03, 0.03, 0.94]])
        obs = np.array([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]])

    max_iter = 100

    # Warm up the numba JIT so compilation time is not charged to the port.
    H.optimize(H.H2mmModel(prior.copy(), trans.copy(), obs.copy()), data, max_iter=2, tol=0.0)

    t0 = time.perf_counter()
    mine = H.optimize(
        H.H2mmModel(prior.copy(), trans.copy(), obs.copy()),
        data, max_iter=max_iter, tol=1e-12,
    )
    t_mine = time.perf_counter() - t0

    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        ref = h2mm_c.EM_H2MM_C(
            h2mm_c.h2mm_model(prior.copy(), trans.copy(), obs.copy()),
            ref_idx, ref_times, max_iter=max_iter,
        )
        t_ref = time.perf_counter() - t0

    mine_ms_it = 1e3 * t_mine / max(mine.n_iter, 1)
    ref_ms_it = 1e3 * t_ref / max(int(ref.niter), 1)
    ratio = mine_ms_it / ref_ms_it

    with capsys.disabled():
        print(
            f"\n[H2MM bench {n_states}-state] N={data.n_photons} "
            f"bursts={data.n_bursts} uniq_dt={int(data.unique_dt.shape[0])}\n"
            f"  numba : {mine_ms_it:7.2f} ms/iter ({mine.n_iter} it)\n"
            f"  H2MM_C: {ref_ms_it:7.2f} ms/iter ({int(ref.niter)} it)\n"
            f"  ratio numba/cpp = {ratio:.2f}x"
        )

    # A ≡ B: a fast but wrong engine would make the timing meaningless.  Loose
    # tolerance — full-precision equivalence is covered by the tests above; this
    # is only a "not garbage" sanity check at a truncated iteration count.
    assert mine.loglik == pytest.approx(ref.loglik, rel=1e-4, abs=1.0)

    # Performance guard: stay under ~1.5x the reference wall-time per iteration.
    # Measured ~0.3-0.6x on dense bursts; the wide margin keeps this robust to
    # CI machine variance while still catching an O(N·n⁴)-style regression.
    assert ratio < 1.5, (
        f"numba H2MM {ratio:.2f}x the H2MM_C time/iter — performance regression"
    )
