"""Image and mosaic generation logic for TTTR Image Browser."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any

import numpy as np

try:
    import tttrlib
except Exception:
    tttrlib = None

_log = logging.getLogger(__name__)

CACHE_DIR_NAME = ".tttr_image_cache"
CACHE_VERSION = "1"


def cache_dir_for(file_path: pathlib.Path) -> pathlib.Path:
    """Return the image-cache directory for *file_path*."""
    d = file_path.parent / CACHE_DIR_NAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def channels_signature(channels_map: dict[str, list[dict]]) -> list[dict]:
    """Build a compact signature of channels_map for hashing (order-independent per detector)."""
    sig: list[dict] = []
    try:
        if not isinstance(channels_map, dict):
            return sig
        for det_name, entries in sorted(channels_map.items(), key=lambda kv: str(kv[0])):
            part = {
                "name": str(det_name),
                "entries": []
            }
            for e in entries or []:
                chs = e.get("detector_chs") or []
                try:
                    chs = sorted([int(c) for c in chs])
                except Exception:
                    pass
                mtr = e.get("micro_time_range") or None
                if isinstance(mtr, (list, tuple)) and len(mtr) == 2:
                    try:
                        mtr = [int(mtr[0]), int(mtr[1])]
                    except Exception:
                        mtr = list(mtr)
                else:
                    mtr = None
                part["entries"].append({"chs": chs, "mtr": mtr})
            part["entries"].sort(key=lambda d: (tuple(d.get("chs") or []), tuple(d.get("mtr") or [])))
            sig.append(part)
    except Exception:
        return sig
    return sig


def mosaic_hash(file_path: pathlib.Path, channels_map: dict[str, list[dict]], reading_routine: str | None) -> str:
    """Return a stable cache signature for a mosaic image load."""
    try:
        st = file_path.stat()
        payload = {
            "v": CACHE_VERSION,
            "file": str(file_path.name),
            "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
            "reading": str(reading_routine) if reading_routine is not None else "Auto",
            "channels": channels_signature(channels_map),
        }
        s = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    except Exception:
        fallback_str = f"{file_path}_{reading_routine}_{json.dumps(channels_signature(channels_map))}"
        return hashlib.sha1(fallback_str.encode("utf-8")).hexdigest()[:16]


def mosaic_cache_file(file_path: pathlib.Path, channels_map: dict[str, list[dict]], reading_routine: str | None) -> pathlib.Path:
    """Return the on-disk cache file path for a mosaic."""
    key = mosaic_hash(file_path, channels_map, reading_routine)
    return cache_dir_for(file_path) / f"{file_path.stem}_mosaic_{key}.npz"


def save_mosaic_cache(
    file_path: pathlib.Path,
    channels_map: dict[str, list[dict]],
    reading_routine: str | None,
    mosaic: np.ndarray,
    labels: list[str],
    cols: int,
    rows: int,
    tile_w: int,
    tile_h: int
) -> None:
    """Save a computed mosaic to disk cache."""
    try:
        cache_path = mosaic_cache_file(file_path, channels_map, reading_routine)
        lab_arr = np.array(labels, dtype=object)
        np.savez_compressed(
            str(cache_path),
            mosaic=mosaic,
            labels=lab_arr,
            cols=int(cols),
            rows=int(rows),
            tile_w=int(tile_w),
            tile_h=int(tile_h)
        )
    except Exception:
        pass


def load_mosaic_cache(
    file_path: pathlib.Path,
    channels_map: dict[str, list[dict]],
    reading_routine: str | None
) -> tuple[np.ndarray, list[str], int, int, int, int] | None:
    """Load a cached mosaic if present."""
    try:
        cache_path = mosaic_cache_file(file_path, channels_map, reading_routine)
        if not cache_path.exists():
            return None
        with np.load(str(cache_path), allow_pickle=True) as z:
            mosaic = z["mosaic"]
            labels = [str(x) for x in z["labels"].tolist()]
            cols = int(z["cols"])
            rows = int(z["rows"])
            tile_w = int(z["tile_w"])
            tile_h = int(z["tile_h"])
            return mosaic, labels, cols, rows, tile_w, tile_h
    except Exception:
        return None


def entry_hash(file_path: pathlib.Path, det_chs: list[int] | None, mtr: list[int] | None, reading_routine: str | None) -> str:
    """Return a signature for a single detector entry cache."""
    try:
        st = file_path.stat()
        payload = {
            "v": CACHE_VERSION,
            "file": str(file_path.name),
            "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
            "reading": str(reading_routine) if reading_routine is not None else "Auto",
            "chs": list(map(int, det_chs)) if det_chs else [],
            "mtr": [int(mtr[0]), int(mtr[1])] if (isinstance(mtr, (list, tuple)) and len(mtr) == 2) else None,
        }
        s = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    except Exception:
        fallback_str = f"{file_path}_{det_chs}_{mtr}_{reading_routine}"
        return hashlib.sha1(fallback_str.encode("utf-8")).hexdigest()[:16]


def entry_cache_file(file_path: pathlib.Path, det_chs: list[int] | None, mtr: list[int] | None, reading_routine: str | None) -> pathlib.Path:
    """Return the entry npz cache file path."""
    key = entry_hash(file_path, det_chs, mtr, reading_routine)
    return cache_dir_for(file_path) / f"{file_path.stem}_{key}.npz"


def compute_entry_stack_cached(
    tttr_obj: Any,
    file_path: pathlib.Path,
    det_chs: list[int] | None,
    mtr: list[int] | None,
    reading_routine: str | None
) -> np.ndarray:
    """Compute and cache a 3D stack (frames, x, y) for an entry."""
    cache_path = entry_cache_file(file_path, det_chs, mtr, reading_routine)
    try:
        if cache_path.exists():
            with np.load(str(cache_path)) as z:
                return z["stack"]
    except Exception:
        pass

    params: dict[str, Any] = {"tttr_data": tttr_obj, "fill": True}
    if det_chs:
        params["channels"] = list(det_chs)
    if mtr is not None and isinstance(mtr, (list, tuple)) and len(mtr) == 2:
        try:
            a, b = int(mtr[0]), int(mtr[1])
            params["micro_time_ranges"] = [(a, b)]
        except Exception:
            pass

    clsm = tttrlib.CLSMImage(**params)
    img = np.array(clsm.intensity)
    if img.ndim == 2:
        img3 = img[None, ...]
    elif img.ndim == 3:
        img3 = img
    else:
        img3 = np.zeros((1, 1, 1), dtype=img.dtype)

    try:
        np.savez_compressed(str(cache_path), stack=img3)
    except Exception:
        pass
    return img3


def sum_stacks_with_padding(stacks: list[np.ndarray]) -> np.ndarray | None:
    """Sum a list of 3D image stacks, applying zero padding if dimensions mismatch."""
    if not stacks:
        return None
    max_x = 0
    max_y = 0
    max_f = 0
    for s in stacks:
        if s.ndim != 3:
            continue
        f, x, y = s.shape
        max_f = max(max_f, f)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if max_x == 0 or max_y == 0:
        return None
    acc = np.zeros((max_f, max_x, max_y), dtype=np.float32)
    for s in stacks:
        if s.ndim != 3:
            continue
        f, x, y = s.shape
        acc[:f, :x, :y] += s.astype(np.float32, copy=False)
    return acc


def get_combo_stack(
    tttr_obj: Any,
    file_path: pathlib.Path,
    entries: list[dict],
    reading_routine: str | None
) -> np.ndarray | None:
    """Sum stacks across multiple detector wizard entries for a combo."""
    stacks = []
    for e in entries:
        det_chs = e.get("detector_chs") or None
        mtr = e.get("micro_time_range") or None
        try:
            st = compute_entry_stack_cached(tttr_obj, file_path, det_chs, mtr, reading_routine)
            stacks.append(st)
        except Exception:
            continue
    return sum_stacks_with_padding(stacks)


def is_clsm_compatible(tttr_obj: Any) -> bool:
    """Check if the TTTR file has valid CLSM headers."""
    try:
        clsm = tttrlib.CLSMImage(tttr_data=tttr_obj)
        _ = getattr(clsm, "intensity", None)
        return _ is not None
    except Exception:
        return False


def get_magma_lut(n: int = 256) -> np.ndarray | None:
    """Load the Magma colormap lookup table."""
    try:
        import matplotlib.cm as cm
        m = cm.get_cmap("magma")
        arr = (m(np.linspace(0, 1, max(2, int(n))))[:, :3] * 255).astype(np.uint8)
        return arr
    except Exception:
        return None


def group_channels_by_detector(channels_map: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Collapse window_detector combos into detector-only combos."""
    grouped: dict[str, list[dict]] = {}
    try:
        for key, entries in (channels_map or {}).items():
            if not isinstance(key, str):
                det_name = str(key)
            else:
                parts = key.split("_", 1)
                det_name = parts[1] if len(parts) > 1 else parts[0]
            lst = grouped.setdefault(det_name, [])
            if isinstance(entries, list):
                lst.extend(entries)
    except Exception:
        return channels_map
    return grouped


def resize_nn(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fast nearest-neighbor 2D array resize."""
    if a.shape == shape:
        return a
    sx = shape[0] / a.shape[0]
    sy = shape[1] / a.shape[1]
    xi = np.clip((np.floor(np.arange(shape[0]) / sx)).astype(int), 0, a.shape[0] - 1)
    yi = np.clip((np.floor(np.arange(shape[1]) / sy)).astype(int), 0, a.shape[1] - 1)
    return a[np.ix_(xi, yi)]


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize and clip array to uint8 range [0, 255]."""
    a = arr.astype(float)
    if not np.isfinite(a).any():
        return np.zeros_like(a, dtype=np.uint8)
    mn, mx = np.nanmin(a), np.nanmax(a)
    if mx <= mn:
        return np.zeros(a.shape, dtype=np.uint8)
    a = (a - mn) / (mx - mn)
    return (a * 255.0).clip(0, 255).astype(np.uint8)


def render_mosaic_array(
    path: pathlib.Path,
    channels_map: dict[str, list[dict]],
    reading_routine: str | None,
    max_side: int = 512
) -> tuple[np.ndarray, list[str], int, int, int, int] | None:
    """Render the mosaic array without Qt dependencies."""
    if tttrlib is None:
        return None
    try:
        tttr_obj = tttrlib.TTTR(str(path), reading_routine)
        if not is_clsm_compatible(tttr_obj):
            return None
    except Exception:
        return None

    names = list(channels_map.keys())
    entries_list = [channels_map[n] for n in names]
    images: list[tuple[str, np.ndarray | None]] = []
    labels: list[str] = []

    for name, entries in zip(names, entries_list):
        stack = get_combo_stack(tttr_obj, path, entries, reading_routine)
        img2d = stack.sum(axis=0) if stack is not None else None
        images.append((name, img2d))

        # Format label name | mt: ranges | ch: routing channels
        try:
            det_chs_set = set()
            for e in entries:
                chs = e.get("detector_chs") or []
                try:
                    det_chs_set.update(int(c) for c in chs)
                except Exception:
                    det_chs_set.update(chs)
            det_chs = sorted(det_chs_set) if det_chs_set else []
            ch_txt = ",".join(map(str, det_chs)) if det_chs else ""

            mtr_list = []
            for e in entries:
                mtr = e.get("micro_time_range") or None
                if isinstance(mtr, (list, tuple)) and len(mtr) == 2 and mtr[0] is not None and mtr[1] is not None:
                    try:
                        mtr_list.append((int(mtr[0]), int(mtr[1])))
                    except Exception:
                        pass
            seen = set()
            uniq_mtrs = []
            for ab in mtr_list:
                if ab not in seen:
                    seen.add(ab)
                    uniq_mtrs.append(ab)
            mtr_txt = ";".join(f"{a}-{b}" for (a, b) in uniq_mtrs)

            parts = [str(name)]
            if mtr_txt:
                parts.append(f"mt: {mtr_txt}")
            if ch_txt:
                parts.append(f"ch: {ch_txt}")
            label_str = "  |  ".join(parts)
        except Exception:
            label_str = str(name)
        labels.append(label_str)

    non_none = [im for _, im in images if im is not None]
    if not non_none:
        return None

    k = len(images)
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    shapes = [(im.shape if im is not None else (1, 1)) for _, im in images]
    max_nx = max(s[0] for s in shapes)
    max_ny = max(s[1] for s in shapes)

    tile_h, tile_w = max_nx, max_ny
    mosaic_h = rows * tile_h
    mosaic_w = cols * tile_w
    scale = 1.0
    if max(mosaic_h, mosaic_w) > max_side:
        scale = max_side / float(max(mosaic_h, mosaic_w))

    tile_h_scaled = max(1, int(round(tile_h * scale)))
    tile_w_scaled = max(1, int(round(tile_w * scale)))
    mosaic = np.zeros((rows * tile_h_scaled, cols * tile_w_scaled), dtype=np.uint8)

    for idx, (_, img) in enumerate(images):
        r, c = divmod(idx, cols)
        y0 = r * tile_h_scaled
        x0 = c * tile_w_scaled
        if img is None:
            tile = np.zeros((tile_h_scaled, tile_w_scaled), dtype=np.uint8)
        else:
            tile = to_uint8(img)
            tile = resize_nn(tile, (tile_h, tile_w))
            if scale != 1.0:
                tile = resize_nn(tile, (tile_h_scaled, tile_w_scaled))
        if tile.shape != (tile_h_scaled, tile_w_scaled):
            tile = resize_nn(tile, (tile_h_scaled, tile_w_scaled))
        mosaic[y0 : y0 + tile_h_scaled, x0 : x0 + tile_w_scaled] = tile

    return mosaic, labels, cols, rows, tile_w_scaled, tile_h_scaled


def load_image(
    path: str,
    setup_settings: dict[str, Any] | None = None,
    max_side: int = 512,
    cache_folder: str | None = None
) -> dict[str, Any] | None:
    """Load or compute a TTTR image mosaic."""
    file_path = pathlib.Path(path)
    reading_routine = None
    if isinstance(setup_settings, dict):
        reading = setup_settings.get("tttr_reading", {})
        reading_routine = reading.get("file_type") or None

    # Group channels map
    try:
        if isinstance(setup_settings, dict) and "detectors" in setup_settings:
            # Reconstruct from DetectorWizard format if available
            channels_map = setup_settings.get("channels", {})
        else:
            channels_map = {}
        # Fallback to default if no channels map
        if not channels_map:
            channels_map = {"Image": [{"window_range": (None, None), "detector_chs": [], "micro_time_range": (None, None)}]}
        channels_map = group_channels_by_detector(channels_map)
    except Exception:
        channels_map = {"Image": [{"window_range": (None, None), "detector_chs": [], "micro_time_range": (None, None)}]}

    # Try cache first
    cached = load_mosaic_cache(file_path, channels_map, reading_routine)
    if cached is not None:
        mosaic, labels, cols, rows, tile_w, tile_h = cached
    else:
        # Precompute
        rendered = render_mosaic_array(file_path, channels_map, reading_routine, max_side=max_side)
        if rendered is None:
            return None
        mosaic, labels, cols, rows, tile_w, tile_h = rendered
        save_mosaic_cache(file_path, channels_map, reading_routine, mosaic, labels, cols, rows, tile_w, tile_h)

    return {
        "path": str(file_path),
        "mosaic": mosaic.tolist(),
        "labels": labels,
        "cols": cols,
        "rows": rows,
    }


def save_tiff_stacks(
    paths: list[str],
    output_dir: str,
    setup_settings: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Export intensity images (per combo) as TIFF stacks."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    reading_routine = None
    if isinstance(setup_settings, dict):
        reading = setup_settings.get("tttr_reading", {})
        reading_routine = reading.get("file_type") or None

    try:
        # Resolve channels map
        if isinstance(setup_settings, dict) and "channels" in setup_settings:
            channels_map = setup_settings.get("channels", {})
        else:
            channels_map = {}
        if not channels_map:
            channels_map = {"Image": [{"window_range": (None, None), "detector_chs": [], "micro_time_range": (None, None)}]}
        channels_map = group_channels_by_detector(channels_map)
    except Exception:
        channels_map = {"Image": [{"window_range": (None, None), "detector_chs": [], "micro_time_range": (None, None)}]}

    for path_str in paths:
        p = pathlib.Path(path_str)
        try:
            tttr_obj = tttrlib.TTTR(str(p), reading_routine)
            for combo_name, entries in channels_map.items():
                stack = get_combo_stack(tttr_obj, p, entries, reading_routine)
                if stack is None:
                    continue
                safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(combo_name))
                dest = out / f"{p.stem}_{safe_name}.tiff"
                ok = False
                try:
                    import tifffile as tiff
                    tiff.imwrite(str(dest), stack.astype(np.uint32, copy=False))
                    ok = True
                except Exception:
                    try:
                        import imageio
                        imageio.mimwrite(str(dest), [frame for frame in stack.astype(np.uint16, copy=False)], format="TIFF")
                        ok = True
                    except Exception:
                        pass
                if ok:
                    saved.append(str(dest))
        except Exception as e:
            _log.warning(f"Failed to save TIFF for {p}: {e}")

    return {"paths": saved}
