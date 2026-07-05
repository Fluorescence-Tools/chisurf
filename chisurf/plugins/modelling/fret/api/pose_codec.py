"""Compact, self-describing codec for docked rigid-body poses.

FPS stores a docked state compactly as the rigid-body *transform vectors*
(a translation + a rotation per body) rather than full atomic coordinates.
This module does the same for ChiSurf docking projects: it packs the pose list
into a single string blob so a whole docked assembly is a handful of numbers
that reconstruct exactly against the same input PDBs.

The blob is a small JSON object that carries its own ``codec`` tag, so it always
decodes without external context:

* ``msgpack+b64`` — ``msgpack.packb(payload)`` base64-encoded (compact, default
  when the ``msgpack`` package is available).
* ``json+b64``    — compact ``json.dumps(payload)`` UTF-8 base64-encoded
  (dependency-free fallback).

Base64 is used rather than hex so the blob is ~33 % overhead instead of 100 %.

A *pose* is a plain dict ``{"body_id": int, "t": [x, y, z], "q": [w, x, y, z]}``
where ``t`` is the reference-frame translation and ``q`` the rotation quaternion.
This module is intentionally free of any IMP import so it can be exercised
without the compiled stack (see :mod:`...core.imp_engine` for pose capture/apply).
"""

from __future__ import annotations

import base64
import json
from typing import Dict, List, Optional

try:  # optional — compact binary encoding
    import msgpack  # type: ignore
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    msgpack = None  # type: ignore

#: Codec used when ``msgpack`` is importable.
_MSGPACK_CODEC = "msgpack+b64"
#: Dependency-free fallback codec.
_JSON_CODEC = "json+b64"

#: Payload schema version (bumped if the pose dict layout changes).
POSE_PAYLOAD_VERSION = 1


def encode_poses(
    poses: List[Dict],
    *,
    score: Optional[float] = None,
    method: str = "",
    frame: Optional[int] = None,
) -> Dict:
    """Pack a list of poses into a compact, self-describing blob dict.

    Parameters
    ----------
    poses : list of dict
        Each ``{"body_id": int, "t": [x, y, z], "q": [w, x, y, z]}``.
    score : float, optional
        Total restraint score of the docked state (for display / provenance).
    method : str
        Docking method that produced the poses (``"minimize"`` / ``"mc"``).
    frame : int, optional
        Iteration / frame index the poses were captured at.

    Returns
    -------
    dict
        ``{"codec": ..., "data": "<hex>", "score": score}`` — JSON-serialisable.
    """
    payload = {
        "v": POSE_PAYLOAD_VERSION,
        "bodies": poses,
        "score": score,
        "method": method,
        "frame": frame,
    }
    if msgpack is not None:
        raw = msgpack.packb(payload, use_bin_type=True)
        return {"codec": _MSGPACK_CODEC, "data": _b64(raw), "score": score}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return {"codec": _JSON_CODEC, "data": _b64(raw), "score": score}


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def decode_poses(blob: Dict) -> Dict:
    """Unpack a blob produced by :func:`encode_poses`.

    Parameters
    ----------
    blob : dict
        ``{"codec": ..., "data": "<hex>", ...}``.

    Returns
    -------
    dict
        The decoded payload: ``{"v", "bodies", "score", "method", "frame"}``.

    Raises
    ------
    ValueError
        If the codec is unknown, or a ``msgpack+hex`` blob is met without the
        ``msgpack`` package installed.
    """
    codec = blob.get("codec")
    raw = base64.b64decode(blob.get("data", ""))
    if codec == _MSGPACK_CODEC:
        if msgpack is None:
            raise ValueError(
                "poses were saved with msgpack but the 'msgpack' package is not "
                "installed; install msgpack to load this project"
            )
        return msgpack.unpackb(raw, raw=False)
    if codec == _JSON_CODEC:
        return json.loads(raw.decode("utf-8"))
    raise ValueError(f"unknown pose codec: {codec!r}")


__all__ = ["encode_poses", "decode_poses", "POSE_PAYLOAD_VERSION"]
