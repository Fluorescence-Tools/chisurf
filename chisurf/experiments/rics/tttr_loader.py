from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pathlib

import numpy as np

try:  # tttrlib is expected to be available in the chisurf dev environment
    import tttrlib  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    tttrlib = None  # type: ignore[var-annotated]


def _guess_tttr_type(path: pathlib.Path) -> Optional[str]:
    """Guess TTTR container type from file extension.

    Returns values understood by :class:`tttrlib.TTTR`, e.g. "PTU" or "HT3".
    """

    suffix = path.suffix.lower()
    if suffix == ".ptu":
        return "PTU"
    if suffix in (".ht3", ".ht2"):
        return "HT3"
    if suffix in (".t3r", ".t2r"):
        return "SPC"
    return None


def load_clsm_from_tttr(
    filename: str | pathlib.Path,
    tttr_type: Optional[str] = None,
    channels: Sequence[int] | None = None,
    reading_routine: str = "default",
    fill: bool = True,
    split_by_channel: bool = False,
    **kwargs: Any,
) -> Tuple["tttrlib.CLSMImage", np.ndarray, Dict[str, Any]]:
    """Load a CLSMImage and intensity stack from a TTTR file using tttrlib.

    This is a thin convenience wrapper around :class:`tttrlib.TTTR` and
    :class:`tttrlib.CLSMImage`, following the patterns used in
    ``modules/tttrlib/examples/flim/plot_read_clsm_data.py`` and
    ``modules/tttrlib/examples/image_correlation/plot_imaging_ics_tttrlib.py``.

    Parameters
    ----------
    filename:
        Path to the TTTR file.
    tttr_type:
        Optional explicit TTTR container type (e.g. "PTU", "HT3").
        If omitted, it is guessed from the file extension.
    channels:
        Sequence of detector channels to include. Defaults to ``[0]``.
    reading_routine:
        tttrlib CLSM reading routine ("default", "SP5", "SP8", ...).
    fill:
        Whether tttrlib should fill the image with photons on construction.
    split_by_channel:
        If *True* and multiple channels are used, the intensity array is
        returned as ``(n_channels, frames_per_channel, n_lines, n_pixel)``.

    Returns
    -------
    clsm:
        The constructed :class:`tttrlib.CLSMImage`.
    intensity:
        Intensity array as a NumPy ndarray. Shape follows
        :pyattr:`tttrlib.CLSMImage.intensity`:

        - single-channel: ``(n_frames, n_lines, n_pixel)``
        - multi-channel with ``split_by_channel=True``:
          ``(n_channels, frames_per_channel, n_lines, n_pixel)``.
    meta:
        Dictionary with basic image metadata where available
        (see :pyfunc:`tttrlib.CLSMImage.get_image_info`).
    """

    if tttrlib is None:
        raise RuntimeError("tttrlib is not available; cannot load CLSM data")

    path = pathlib.Path(filename)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if tttr_type is None:
        tttr_type = _guess_tttr_type(path)

    if tttr_type is None:
        # Let tttrlib auto-detect type from header if possible
        tttr = tttrlib.TTTR(str(path))  # type: ignore[call-arg]
    else:
        tttr = tttrlib.TTTR(str(path), tttr_type)  # type: ignore[call-arg]

    if channels is None:
        channels = (0,)

    reading_routine = str(reading_routine or "default")

    clsm = tttrlib.CLSMImage(  # type: ignore[call-arg]
        tttr_data=tttr,
        channels=list(channels),
        fill=bool(fill),
        reading_routine=reading_routine,
        split_by_channel=bool(split_by_channel),
        **kwargs,
    )

    intensity = np.asarray(clsm.intensity, dtype=float)

    meta: Dict[str, Any] = {}
    try:
        if hasattr(clsm, "get_image_info"):
            info = clsm.get_image_info()  # type: ignore[assignment]
            if isinstance(info, dict):
                meta.update(info)
    except Exception:
        # Metadata is optional; ignore failures.
        pass

    # Basic shape metadata that is always available
    try:
        shape = getattr(clsm, "shape", None)
        if shape is not None:
            meta.setdefault("clsm_shape", tuple(shape))
    except Exception:
        pass

    return clsm, intensity, meta
