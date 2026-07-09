"""H2MM compute-engine selection — one dispatcher over the fitting back-ends.

The plugin offers several ways to estimate an H2MM model at a given state count;
they trade exactness for speed. This module is the single place that maps an
``engine`` name to the corresponding call, so the CLI, GUI, backend service, and
:func:`~.analysis.analyze` all agree.

Engines
-------
``em``
    Exact Baum-Welch EM with SQUAREM acceleration (the default). Reproduces the
    reference maximum-likelihood estimate.
``em-float32``
    The same EM in the approximate ``float32`` fast mode (~1.5–2× faster, small
    round-off; see :func:`~.h2mm.optimize`).
``surrogate``
    The optional amortised neural estimator (:mod:`.surrogate`): one forward pass,
    **approximate**, ~5–25× faster. Requires a trained surrogate for the state
    count; falls back to ``em`` where none is supplied.
``surrogate-refine``
    ``surrogate`` seed polished by ``refine_iters`` Baum-Welch maps.

For model selection over several state counts, the *scan* always scores fitted
models by BIC/ICL as usual; only *how each model is fitted* changes.
"""

from __future__ import annotations

from .h2mm import BurstPhotons, H2mmModel, fit_states

ENGINES: tuple[str, ...] = ("em", "em-float32", "surrogate", "surrogate-refine")

ENGINE_LABELS: dict[str, str] = {
    "em": "EM (exact)",
    "em-float32": "EM float32 (fast, approximate)",
    "surrogate": "Surrogate NN (fastest, approximate)",
    "surrogate-refine": "Surrogate NN + EM polish",
}


def normalize_engine(engine: str | None) -> str:
    """Return a valid engine name, defaulting unknown/empty values to ``em``."""
    e = (engine or "em").strip().lower()
    return e if e in ENGINES else "em"


def fit_one(
    data: BurstPhotons,
    n_states: int,
    engine: str = "em",
    *,
    surrogates: dict[int, object] | None = None,
    refine_iters: int = 20,
    n_restarts: int = 2,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int = 0,
) -> H2mmModel:
    """Fit a single ``n_states`` model with the selected ``engine``.

    Parameters
    ----------
    data : BurstPhotons
        Photon data in engine layout.
    n_states : int
        State count to fit.
    engine : str
        One of :data:`ENGINES`.
    surrogates : dict, optional
        Mapping ``n_states -> SurrogateModel`` (or path). Used by the surrogate
        engines; missing entries fall back to exact EM.
    refine_iters : int
        EM polish maps for ``surrogate-refine``.
    n_restarts, max_iter, tol, seed
        EM parameters (passed to :func:`~.h2mm.fit_states`).

    Returns
    -------
    H2mmModel
        The fitted model.
    """
    engine = normalize_engine(engine)

    if engine in ("surrogate", "surrogate-refine"):
        sm = (surrogates or {}).get(int(n_states))
        if sm is not None:
            ri = int(refine_iters) if engine == "surrogate-refine" else 0
            return fit_states(data, n_states, surrogate=sm, refine_iters=ri, tol=tol)
        # No surrogate for this state count → exact EM keeps the scan usable.
        engine = "em"

    return fit_states(
        data,
        n_states,
        n_restarts=n_restarts,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        single_precision=(engine == "em-float32"),
    )
