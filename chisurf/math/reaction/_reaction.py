"""Numba-based stochastic reaction model (Gillespie SSA).

This module replaces the legacy Cython extension
``chisurf.math.reaction.reaction_`` with a pure-Python/Numba
implementation. It provides a small :class:`Model` API compatible
with the original example usage in :mod:`chisurf.math.reaction.stochastic`.

The implementation intentionally mirrors the structure of the old
Cython class:

- ``Model`` holds variable names, rates, initial values, transition
  matrix, and a list of propensity functions.
- ``run()`` allocates an output array and repeatedly calls ``GSSA``.
- ``GSSA`` implements the Gillespie Direct method using NumPy and an
  internal helper that is optionally accelerated with Numba.

The propensity functions are still plain Python callables of the
form ``f(rates, state) -> float``. Because of this, the Numba kernel
runs in object mode (``forceobj=True``) when Numba is available. If
Numba is not available at runtime, a pure-Python fallback with the
same semantics is used automatically.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.random import multinomial

try:  # Optional acceleration via numba
    import numba as nb  # type: ignore[import]

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - runtime availability only
    nb = None  # type: ignore[assignment]
    _HAVE_NUMBA = False


DTYPE = np.float64


if _HAVE_NUMBA:

    @nb.jit(forceobj=True)  # uses Python callables in the loop
    def _gssa_loop(
        tmax: int,
        ini: np.ndarray,
        rates: np.ndarray,
        pv_funcs: Sequence[Callable[[np.ndarray, np.ndarray], float]],
        tm: np.ndarray,
        res: np.ndarray,
        round_idx: int,
    ) -> int:
        """Inner Gillespie SSA loop (Numba-accelerated, object mode).

        Parameters are deliberately kept generic so the semantics match
        the original Cython implementation.
        """

        l = len(pv_funcs)
        pv = np.zeros(l, dtype=DTYPE)

        tc = 0.0
        steps = 0
        a0 = 1.0

        # initial state at t = 0
        res[0, :, round_idx] = ini

        for tim in range(1, tmax):
            while tc < tim:
                # compute propensity vector
                for i in range(l):
                    pv[i] = pv_funcs[i](rates, ini)

                a0 = float(np.sum(pv))
                if a0 <= 0.0:
                    break

                tau = (-1.0 / a0) * float(np.log(np.random.random()))
                probs = pv / a0
                event = multinomial(1, probs)
                idx = int(np.nonzero(event)[0][0])

                ini = ini + tm[:, idx]
                tc += tau
                steps += 1

            res[tim, :, round_idx] = ini
            if a0 <= 0.0:
                break

        return steps


else:

    def _gssa_loop(
        tmax: int,
        ini: np.ndarray,
        rates: np.ndarray,
        pv_funcs: Sequence[Callable[[np.ndarray, np.ndarray], float]],
        tm: np.ndarray,
        res: np.ndarray,
        round_idx: int,
    ) -> int:
        """Pure-Python Gillespie SSA loop (no Numba available)."""

        l = len(pv_funcs)
        pv = np.zeros(l, dtype=DTYPE)

        tc = 0.0
        steps = 0
        a0 = 1.0

        res[0, :, round_idx] = ini

        for tim in range(1, tmax):
            while tc < tim:
                for i in range(l):
                    pv[i] = pv_funcs[i](rates, ini)

                a0 = float(np.sum(pv))
                if a0 <= 0.0:
                    break

                tau = (-1.0 / a0) * float(np.log(np.random.random()))
                probs = pv / a0
                event = multinomial(1, probs)
                idx = int(np.nonzero(event)[0][0])

                ini = ini + tm[:, idx]
                tc += tau
                steps += 1

            res[tim, :, round_idx] = ini
            if a0 <= 0.0:
                break

        return steps


class Model:
    """Stochastic reaction model using the Gillespie SSA algorithm.

    Parameters
    ----------
    variable_names:
        Names of the state variables.
    rate_constants:
        Array-like of rate constants.
    inits:
        Initial state vector.
    transition_matrix:
        Stoichiometric transition matrix with shape (n_vars, n_reactions).
    propensity:
        Sequence of callables ``f(rates, state) -> float`` providing the
        propensities for each reaction channel.
    """

    def __init__(
        self,
        vnames: Sequence[str],
        rates: Iterable[float],
        inits: Iterable[float],
        tmat: np.ndarray,
        propensity: Sequence[Callable[[np.ndarray, np.ndarray], float]],
    ) -> None:
        # keep legacy attribute names from the Cython implementation
        self.vn: List[str] = list(vnames)
        self.rates = np.asarray(rates, dtype=DTYPE)
        self.inits = np.asarray(inits, dtype=DTYPE)
        self.tm = np.asarray(tmat, dtype=DTYPE)
        self.pv = list(propensity)

        self.pvl = len(self.pv)
        self.nvars = int(self.inits.shape[0])

        self.time = np.zeros(1, dtype=DTYPE)
        self.series = np.zeros(1, dtype=DTYPE)
        self.steps = 0
        self.res: np.ndarray | None = None

    def run(
        self,
        method: str = "SSA",
        tmax: int = 10,
        reps: int = 1,
    ) -> None:
        """Run the stochastic simulation.

        Only ``method='SSA'`` is currently implemented, matching the
        legacy Cython backend.
        """

        res = np.zeros((tmax, self.nvars, reps), dtype=DTYPE)
        tvec = np.arange(tmax, dtype=DTYPE)
        self.res = res

        steps = 0
        if method == "SSA":
            for i in range(reps):
                # Use a fresh copy of the initial state per repetition,
                # which is slightly safer than mutating ``self.inits``
                # across runs.
                steps = _gssa_loop(
                    int(tmax),
                    self.inits.copy(),
                    self.rates,
                    self.pv,
                    self.tm,
                    res,
                    int(i),
                )

        self.time = tvec
        self.series = res
        self.steps = int(steps)

    def getStats(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return ``(time, series, steps)`` as in the original Model."""

        if self.res is None:
            raise RuntimeError("Model.run() must be called before getStats().")
        return self.time, self.series, self.steps

    def GSSA(self, tmax: int = 50, round: int = 0) -> int:  # noqa: A003 - keep API name
        """Run a single SSA trajectory into the existing ``res`` array."""

        if self.res is None:
            raise RuntimeError("Model.run() must be called before GSSA().")

        return int(
            _gssa_loop(
                int(tmax),
                self.inits.copy(),
                self.rates,
                self.pv,
                self.tm,
                self.res,
                int(round),
            )
        )

    def CR(self, pv: np.ndarray) -> int:  # placeholder to match old API
        """Composition-reaction algorithm (not yet implemented).

        This is a stub kept for API compatibility with the original
        Cython class. It currently raises ``NotImplementedError``.
        """

        raise NotImplementedError("Composition-reaction (CR) is not implemented.")


def l1(r: np.ndarray, ini: np.ndarray) -> float:
    """Example propensity: bimolecular reaction rate r[0] * x0 * x1."""

    return float(r[0] * ini[0] * ini[1])


def l2(r: np.ndarray, ini: np.ndarray) -> float:
    """Example propensity: unimolecular decay r[1] * x1."""

    return float(r[1] * ini[1])
