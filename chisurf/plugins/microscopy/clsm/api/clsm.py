"""Headless orchestration for the CLSM plugin.

These functions take JSON-safe parameters and return JSON-safe dicts, so they
can be driven identically from the CLI, the RPC services, and tests. ``tttrlib``
and the ``core`` image routines are imported lazily.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from ..core import imaging, setups
from .models import ClsmSetup, DecayResult, FrcResult, RepresentationResult

# ── setup resolution ───────────────────────────────────────────────────────


def list_setups() -> dict[str, Any]:
    """Return the built-in CLSM setup presets."""
    return setups.builtin_setups()


def _resolve_setup(
    filename: str | None = None,
    *,
    setup_name: str | None = None,
    channels: list[int] | None = None,
    auto_detect: bool = True,
    tttr_type: str | None = None,
    frame_marker: list[int] | None = None,
    line_start_marker: int | None = None,
    line_stop_marker: int | None = None,
    event_type_marker: int | None = None,
    pixel_per_line: int | None = None,
    routine: str | None = None,
    tttr: Any = None,
) -> ClsmSetup:
    """Build a :class:`ClsmSetup` from a preset, file markers, and overrides.

    Precedence (low → high): preset → file-detected markers → explicit args.
    """
    presets = setups.builtin_setups()
    base = dict(presets.get(setup_name, {})) if setup_name else {}
    setup = (
        ClsmSetup.from_preset(base, channels=channels)
        if base
        else ClsmSetup(channels=list(channels) if channels is not None else [0])
    )

    if auto_detect and (tttr is not None or filename):
        try:
            import tttrlib

            t = tttr if tttr is not None else tttrlib.TTTR(filename, setup.tttr_type)
            detected = setups.read_clsm_markers(t)
            for key in (
                "frame_marker",
                "line_start_marker",
                "line_stop_marker",
                "event_type_marker",
                "pixel_per_line",
            ):
                if detected.get(key) is not None:
                    setattr(setup, key, detected[key])
        except Exception:
            pass  # detection is best-effort; fall back to preset/defaults

    overrides = {
        "tttr_type": tttr_type,
        "frame_marker": frame_marker,
        "line_start_marker": line_start_marker,
        "line_stop_marker": line_stop_marker,
        "event_type_marker": event_type_marker,
        "pixel_per_line": pixel_per_line,
        "routine": routine,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(setup, key, value)
    if channels is not None:
        setup.channels = list(channels)
    return setup


def _load(filename: str, setup: ClsmSetup) -> tuple[Any, Any]:
    """Load a TTTR file and build the CLSM image for *setup*."""
    import tttrlib

    tttr = tttrlib.TTTR(filename, setup.tttr_type)
    clsm_image = imaging.build_clsm_image(tttr, setup)
    return tttr, clsm_image


# ── operations ─────────────────────────────────────────────────────────────


def image_info(filename: str, **setup_kwargs: Any) -> dict[str, Any]:
    """Load a file, build a CLSM image and report its dimensions and setup."""
    setup = _resolve_setup(filename, **setup_kwargs)
    tttr, clsm_image = _load(filename, setup)
    header = tttr.get_header()
    return {
        "filename": filename,
        "setup": vars(setup),
        "n_frames": int(clsm_image.n_frames),
        "n_lines": int(clsm_image.n_lines),
        "n_pixel": int(clsm_image.n_pixel),
        "n_photons": int(len(tttr)),
        "micro_time_resolution_ns": float(header.micro_time_resolution * 1e9),
    }


def compute_representation(
    filename: str,
    image_type: str = "Intensity",
    n_ph_min: int = 1,
    frame_mode: str = "sum",
    frame_idx: int = 0,
    output_path: str | None = None,
    **setup_kwargs: Any,
) -> dict[str, Any]:
    """Compute an image representation and optionally write it to disk.

    ``output_path`` ending in ``.npy`` stores the full 3-D stack; an image
    extension (``.png``/``.tif``) stores the reduced 2-D frame.
    """
    setup = _resolve_setup(filename, **setup_kwargs)
    tttr, clsm_image = _load(filename, setup)
    image = imaging.representation(clsm_image, tttr, image_type, n_ph_min)
    current, _, _ = imaging.reduce_frames(image, frame_mode, frame_idx)

    saved = ""
    if output_path:
        out = pathlib.Path(output_path)
        if out.suffix == ".npy":
            np.save(out, image)
        else:
            import skimage as ski

            ski.io.imsave(str(out), current)
        saved = str(out)

    result = RepresentationResult(
        image_type=image_type,
        n_frames=int(image.shape[0]),
        n_lines=int(image.shape[1]),
        n_pixel=int(image.shape[2]),
        total_intensity=float(image.sum()),
        output_path=saved,
    )
    return vars(result)


def _selection_mask(
    clsm_image: Any,
    tttr: Any,
    image_type: str,
    n_ph_min: int,
    frame_mode: str,
    frame_idx: int,
    mask_path: str | None,
    threshold: float | None,
) -> np.ndarray:
    """Resolve a 2-D selection mask from a file, a threshold, or all pixels."""
    if mask_path:
        import skimage as ski

        mask = np.asarray(ski.io.imread(mask_path))
        if mask.ndim == 3:
            mask = mask[0]
        return mask
    if threshold is not None:
        image = imaging.representation(clsm_image, tttr, image_type, n_ph_min)
        current, _, _ = imaging.reduce_frames(image, frame_mode, frame_idx)
        return (current > threshold * float(current.max())).astype(np.uint8)
    return np.ones((clsm_image.n_lines, clsm_image.n_pixel), dtype=np.uint8)


def extract_decay(
    filename: str,
    mask_path: str | None = None,
    threshold: float | None = None,
    image_type: str = "Intensity",
    n_ph_min: int = 1,
    frame_mode: str = "sum",
    frame_idx: int = 0,
    tac_coarsening: int = 1,
    stack_frames: bool = True,
    output_path: str | None = None,
    **setup_kwargs: Any,
) -> dict[str, Any]:
    """Extract a decay histogram from a pixel selection and optionally save it.

    The selection is, in order of precedence, *mask_path* (an image file),
    *threshold* (fraction of the representation's max), or every pixel.
    ``output_path`` writes a 3-column ``t<TAB>y<TAB>ey`` text file.
    """
    setup = _resolve_setup(filename, **setup_kwargs)
    tttr, clsm_image = _load(filename, setup)
    mask = _selection_mask(
        clsm_image, tttr, image_type, n_ph_min, frame_mode, frame_idx, mask_path, threshold
    )
    t, y, ey = imaging.decay_of_selection(
        clsm_image,
        tttr,
        mask,
        tac_coarsening=tac_coarsening,
        stack_frames=stack_frames,
        frame_idx=frame_idx,
    )

    saved = ""
    if output_path:
        np.savetxt(
            output_path,
            np.vstack([t, y, ey]).T,
            delimiter="\t",
            header="t_ns\tcounts\tnoise",
            comments="",
        )
        saved = str(output_path)

    result = DecayResult(
        time_ns=t.tolist(),
        counts=y.tolist(),
        noise=ey.tolist(),
        n_photons=int(y.sum()),
        output_path=saved,
    )
    return vars(result)


def compute_frc(
    filename: str,
    image_type: str = "Intensity",
    n_ph_min: int = 1,
    frame_mode: str = "sum",
    frame_idx: int = 0,
    bin_width: float = 2.0,
    output_path: str | None = None,
    **setup_kwargs: Any,
) -> dict[str, Any]:
    """Compute the Fourier Ring Correlation for a file's image representation."""
    from ..core import frc as frc_mod

    setup = _resolve_setup(filename, **setup_kwargs)
    tttr, clsm_image = _load(filename, setup)
    image = imaging.representation(clsm_image, tttr, image_type, n_ph_min)
    _, subset_1, subset_2 = imaging.reduce_frames(image, frame_mode, frame_idx)
    density, bins = frc_mod.compute_frc(subset_1, subset_2, bin_width)

    saved = ""
    if output_path:
        np.savetxt(
            output_path,
            np.vstack([bins, density]).T,
            delimiter="\t",
            header="frequency\tcorrelation",
            comments="",
        )
        saved = str(output_path)

    result = FrcResult(
        frequency=bins.tolist(),
        correlation=np.nan_to_num(density).tolist(),
        output_path=saved,
    )
    return vars(result)
