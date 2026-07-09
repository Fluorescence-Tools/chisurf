"""Generate a small synthetic smFRET dataset for the H2MM examples.

Simulates a two-state single-molecule FRET experiment and writes a **real,
loadable dataset** — a Photon-HDF5 ``tttrlib`` file plus a Seidel-style ``.bur``
burst table that references it — so the H2MM GUI, the ``h2mm`` CLI, and the
tutorial notebook can all run on genuine on-disk example data (no measured file
required).

Run as a script::

    python -m chisurf.plugins.burst.burst_h2mm.examples.generate_example_data --out ./h2mm_example

then analyse the result with::

    h2mm compute ./h2mm_example --file-type auto
"""

from __future__ import annotations

import pathlib

import numpy as np


def default_model():
    """Return the two-state ground-truth model used by the examples."""
    from ..core.h2mm import H2mmModel

    return H2mmModel(
        prior=np.array([0.5, 0.5]),
        trans=np.array([[0.995, 0.005], [0.010, 0.990]]),
        obs=np.array([[0.85, 0.15], [0.20, 0.80]]),  # E ≈ 0.15 and 0.80
    )


def generate_example_data(
    out_dir: str | pathlib.Path,
    n_bursts: int = 400,
    burst_len: int = 120,
    mean_dt: int = 4,
    seed: int = 1,
    model=None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write a synthetic smFRET ``(.bur, .photon.h5)`` example dataset.

    Parameters
    ----------
    out_dir : str or Path
        Directory to write into (created if needed).
    n_bursts, burst_len : int
        Number of bursts and photons per burst.
    mean_dt : int
        Mean inter-photon time (macro-time ticks).
    seed : int
        RNG seed.
    model : H2mmModel, optional
        Generative model; defaults to :func:`default_model`.

    Returns
    -------
    bur_path, tttr_path : pathlib.Path
        The written ``.bur`` burst table and the Photon-HDF5 TTTR file. Load the
        folder with ``file_type="auto"`` (the format is auto-detected).
    """
    import tttrlib

    from ..core import h2mm

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = model or default_model()

    rng = np.random.default_rng(seed)
    times = [
        np.concatenate([[0], np.cumsum(rng.poisson(mean_dt, burst_len - 1) + 1)]).astype(np.int64)
        for _ in range(n_bursts)
    ]
    streams = h2mm.simulate_bursts(gt, times, seed=seed + 7)

    tttr_path = out_dir / "sim_smfret.photon.h5"
    macro, chan, rows = [], [], []
    off, base = 0, 0
    for t, s in zip(times, streams):
        macro.append((t + base).astype(np.uint64))
        chan.append(s.astype(np.int8))                 # channel 0 = donor, 1 = acceptor
        rows.append((tttr_path.name, off, off + len(t)))
        off += len(t)
        base += int(t[-1]) + 100000                    # large gap keeps bursts distinct

    macro = np.concatenate(macro).astype(np.uint64)
    chan = np.concatenate(chan).astype(np.int8)
    micro = np.zeros(macro.size, dtype=np.uint16)
    tttr = tttrlib.TTTR()
    tttr.append_events(macro, micro, chan, np.zeros(macro.size, dtype=np.int8), False, 0)
    tttr.write_hdf_file(str(tttr_path))

    import pandas as pd

    bur_path = out_dir / "sim_smfret.bur"
    pd.DataFrame(rows, columns=["First File", "First Photon", "Last Photon"]).to_csv(
        bur_path, sep="\t", index=False
    )
    return bur_path, tttr_path


def _main() -> None:
    """Command-line entry point (``python -m ...generate_example_data``)."""
    import argparse

    p = argparse.ArgumentParser(description="Generate a synthetic smFRET H2MM example dataset.")
    p.add_argument("--out", default="./h2mm_example", help="Output directory")
    p.add_argument("--bursts", type=int, default=400)
    p.add_argument("--burst-len", type=int, default=120)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    bur, tttr = generate_example_data(
        args.out, n_bursts=args.bursts, burst_len=args.burst_len, seed=args.seed
    )
    print(f"wrote {bur}")
    print(f"wrote {tttr}")
    print(f"analyse with:  h2mm compute {args.out} --file-type auto")


if __name__ == "__main__":
    _main()
