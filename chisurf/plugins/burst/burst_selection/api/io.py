"""Input/output helpers for the Burst Selection API."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import tttrlib

from chisurf.core.fio.fluorescence.burst import read_bur_file, write_dataframe_to_bur


def load_tttr(path: str | Path, filetype: str | None = None) -> tttrlib.TTTR:
    """Load a TTTR file using ``tttrlib``.

    Parameters
    ----------
    path : str or Path
        TTTR file path.
    filetype : str, optional
        Explicit TTTR file type passed to ``tttrlib.TTTR``.

    Returns
    -------
    tttrlib.TTTR
        Loaded TTTR object.
    """
    if filetype:
        return tttrlib.TTTR(str(path), filetype)
    return tttrlib.TTTR(str(path))


def read_bur(path: str | Path) -> pd.DataFrame:
    """Read a ChiSurf ``.bur`` file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the ``.bur`` file.

    Returns
    -------
    pandas.DataFrame
        Burst summary table.
    """
    return read_bur_file(path)


def write_bur(df: pd.DataFrame, path: str | Path) -> None:
    """Write a burst summary DataFrame as a ChiSurf ``.bur`` file.

    Parameters
    ----------
    df : pandas.DataFrame
        Burst summary table.
    path : str or Path
        Output path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe_to_bur(df, target)


def get_unique_folder_path(base_path: Path) -> Path:
    """Return a path unique w.r.t. both the directory and a sibling ``.zip``.

    If *base_path* exists or a sibling ``{base_path.name}.zip`` exists,
    numeric suffixes (``_0``, ``_1``, …) are tried until a free name is found.

    Parameters
    ----------
    base_path : Path
        Desired output directory path.

    Returns
    -------
    Path
        Unique directory path.
    """

    def _name_taken(p: Path) -> bool:
        return p.exists() or (p.parent / f"{p.name}.zip").exists()

    if not _name_taken(base_path):
        return base_path
    counter = 0
    while True:
        candidate = base_path.parent / f"{base_path.name}_{counter}"
        if not _name_taken(candidate):
            return candidate
        counter += 1


def _pick_complib(target_dir: Path) -> str:
    """Pick the best available HDF5 compression library."""
    for candidate in ("bzip2", "blosc", "zlib"):
        tmp = target_dir / "__hdf5_complib_test__.h5"
        try:
            pd.HDFStore(str(tmp), mode="w", complib=candidate).close()
            tmp.unlink(missing_ok=True)
            return candidate
        except Exception:
            tmp.unlink(missing_ok=True)
            continue
    return "zlib"


def _prepare_hdf5_dataframe(combined: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list]]:
    """Apply legacy HDF5 encoding steps: drop all-NaN cols, downcast, encode objects.

    Returns
    -------
    df : pd.DataFrame
        Encoded DataFrame (object columns replaced by ``int32`` codes).
    cat_map : dict[str, list]
        Mapping of column name → category list for decoding.
    """
    # --- drop empty columns
    df = combined.dropna(axis=1, how="all").copy()

    # --- downcast numerics
    for c in df.select_dtypes(include=["integer"]).columns:
        if (df[c] >= 0).all():
            df[c] = pd.to_numeric(df[c], downcast="unsigned")
        else:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["floating"]).columns:
        df[c] = df[c].astype(np.float32)

    # --- encode strings/objects as categorical codes
    cat_map: dict[str, list] = {}
    obj_cols = df.select_dtypes(include=["object"]).columns
    must_encode = {"Source File", "First File", "Last File"} & set(obj_cols)

    for col in obj_cols:
        nunique = df[col].nunique(dropna=False)
        if (col in must_encode) or (nunique <= 0.5 * len(df)):
            cat = pd.Categorical(df[col], ordered=False)
            cat_map[col] = cat.categories.tolist()
            df[col] = cat.codes.astype(np.int32)
        else:
            cat = pd.Categorical(df[col], ordered=False)
            cat_map[col] = cat.categories.tolist()
            df[col] = cat.codes.astype(np.int32)

    return df, cat_map


def write_hdf5(
    dataframes: Sequence[pd.DataFrame],
    path: str | Path,
    complib: str | None = None,
) -> None:
    """Write one or more burst DataFrames as a legacy-compatible HDF5 file.

    The output matches the format produced by
    ``WizardTTTRPhotonFilter.save_selection`` with
    ``"hdf5"`` in *output_types*:

    * All DataFrames are concatenated.
    * All-NaN columns are dropped.
    * Integers are downcast (unsigned where possible).
    * Floats are cast to ``np.float32``.
    * Object columns are encoded as ``int32`` categorical codes; the
      mapping is stored in the HDF5 storer attribute ``category_map``.
    * Written with ``pd.HDFStore``, fixed format, key ``"results"``, no
      index.

    Parameters
    ----------
    dataframes : sequence of pandas.DataFrame
        Burst summary tables to combine.
    path : str or Path
        Output ``.h5`` path.
    complib : str, optional
        Compression library override (default: auto-detect ``bzip2`` →
        ``blosc`` → ``zlib``).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not dataframes:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(list(dataframes), ignore_index=True)
    df, cat_map = _prepare_hdf5_dataframe(combined)

    if complib is None:
        complib = _pick_complib(target.parent)

    with pd.HDFStore(str(target), mode="w", complib=complib, complevel=9) as store:
        store.put("results", df, format="fixed", index=False)
        st = store.get_storer("results")
        st.attrs.category_map = json.dumps(cat_map)


def zip_output_folder(output_folder: Path, zip_path: str | Path | None = None) -> Path:
    """Zip an output folder preserving relative paths, matching legacy format.

    Parameters
    ----------
    output_folder : Path
        Directory to zip.
    zip_path : str or Path, optional
        Desired zip path. If not given, ``{output_folder}.zip`` is
        used (with ``_N`` suffix if that name is taken).

    Returns
    -------
    Path
        Path to the created zip file.
    """
    if not output_folder.exists() or not output_folder.is_dir():
        raise FileNotFoundError(f"Output folder not found: {output_folder}")

    if zip_path is None:
        zip_path = get_unique_folder_path(output_folder).parent / f"{output_folder.name}.zip"

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in sorted(output_folder.rglob("*")):
            if entry.is_file():
                rel = entry.relative_to(output_folder)
                zf.write(entry, str(rel))

    return zip_path
