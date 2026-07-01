"""Lifetime / rotation spectrum persistence for the anisotropy wizard.

The wizard edits two spectra — a fluorescence *lifetime* spectrum and a
*rotation* (anisotropy) spectrum — each a list of ``[amplitude, value]`` pairs.
They are stored as ``*.spk.json`` files in the plugin's user-settings folder.
This module owns the (Qt-free) file layout, defaults and (de)serialisation so the
GUI and CLI share one implementation and it can be unit-tested without a GUI.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import typing

import numpy as np

#: Packaged default spectra (also written when no user file exists yet).
DEFAULT_SPECTRA: dict[str, list] = {
    "lifetime_spectrum": [[0.3, 1.8], [0.7, 4.1]],
    "rotation_spectrum": [[0.28, 0.15], [0.1, 10.0]],
}


def user_settings_dir() -> pathlib.Path:
    """Return (creating) the plugin's user-settings directory."""
    import chisurf.core.settings

    path = chisurf.core.settings.chisurf_settings_path / "plugins" / "tr_anisotropy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def packaged_default_path() -> pathlib.Path:
    """Return the path to the packaged ``wizard.spk.json`` default."""
    return pathlib.Path(__file__).parent.parent / "wizard.spk.json"


def spk_json_path() -> pathlib.Path:
    """Return the user's ``wizard.spk.json`` path, seeding it from defaults."""
    path = user_settings_dir() / "wizard.spk.json"
    if not path.exists():
        default = packaged_default_path()
        if default.exists():
            shutil.copyfile(default, path)
        else:
            path.write_text(json.dumps(DEFAULT_SPECTRA))
    return path


def load_spectra(path: str | pathlib.Path) -> dict[str, list]:
    """Load ``{lifetime_spectrum, rotation_spectrum}`` from a ``*.spk.json`` file."""
    with open(path) as fp:
        data = json.load(fp)
    return {
        "lifetime_spectrum": data.get("lifetime_spectrum", []),
        "rotation_spectrum": data.get("rotation_spectrum", []),
    }


def save_spectra(
    path: str | pathlib.Path,
    lifetime_pairs: typing.Sequence[typing.Sequence[float]],
    rotation_pairs: typing.Sequence[typing.Sequence[float]],
    *,
    mirror_to_default: bool = True,
) -> None:
    """Write both spectra to *path* (and, optionally, to the default location).

    Parameters
    ----------
    path : str or Path
        Destination ``*.spk.json`` file.
    lifetime_pairs, rotation_pairs : sequence of [amplitude, value]
        The spectra to write.
    mirror_to_default : bool, optional
        When *path* differs from the canonical user default, also copy it there
        (backing up the previous default). Mirrors the legacy wizard behaviour.
    """
    path = pathlib.Path(path)
    payload = {
        "lifetime_spectrum": [[float(a), float(b)] for a, b in lifetime_pairs],
        "rotation_spectrum": [[float(a), float(b)] for a, b in rotation_pairs],
    }
    if not path.parent.is_dir():
        return
    with open(path, "w+") as fp:
        json.dump(payload, fp)

    if mirror_to_default:
        default = spk_json_path()
        if path.resolve() != default.resolve():
            if default.exists():
                shutil.copyfile(default, default.with_suffix(".backup.json"))
            shutil.copyfile(path, default)


def flatten(pairs: typing.Sequence[typing.Sequence[float]]) -> np.ndarray:
    """Return a flat ``[a0, v0, a1, v1, …]`` array from ``[[a, v], …]`` pairs."""
    return np.asarray(pairs, dtype=np.float64).flatten()


def to_pairs(flat: typing.Sequence[float]) -> list[list[float]]:
    """Return ``[[a, v], …]`` pairs from a flat ``[a0, v0, a1, v1, …]`` array."""
    flat = list(flat)
    return [[float(flat[i]), float(flat[i + 1])] for i in range(0, len(flat) - 1, 2)]
