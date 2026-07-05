"""Round-trip tests for the compact pose codec (no IMP required)."""

from __future__ import annotations

import math

import pytest

from chisurf.plugins.modelling.fret.api import pose_codec

_POSES = [
    {"body_id": 0, "t": [0.0, 0.0, 0.0], "q": [1.0, 0.0, 0.0, 0.0]},
    {"body_id": 1, "t": [10.5, -20.25, 15.125], "q": [0.5, 0.5, 0.5, 0.5]},
    {"body_id": 3, "t": [-1.234567, 8.9, 0.0001], "q": [0.7071, 0.0, 0.7071, 0.0]},
]


def _assert_round_trips(blob, expect_codec):
    assert blob["codec"] == expect_codec
    assert isinstance(blob["data"], str)
    payload = pose_codec.decode_poses(blob)
    assert payload["method"] == "minimize"
    assert payload["score"] == pytest.approx(42.5)
    bodies = payload["bodies"]
    assert [b["body_id"] for b in bodies] == [0, 1, 3]
    for got, exp in zip(bodies, _POSES):
        assert got["body_id"] == exp["body_id"]
        for a, b in zip(got["t"], exp["t"]):
            assert a == pytest.approx(b, abs=1e-6)
        # quaternion recovered up to numerical precision
        dot = sum(a * b for a, b in zip(got["q"], exp["q"]))
        assert abs(dot) == pytest.approx(1.0, abs=1e-3) or math.isclose(
            sum((a - b) ** 2 for a, b in zip(got["q"], exp["q"])), 0.0, abs_tol=1e-6)


def test_round_trip_default_codec():
    blob = pose_codec.encode_poses(_POSES, score=42.5, method="minimize")
    expect = "msgpack+b64" if pose_codec.msgpack is not None else "json+b64"
    _assert_round_trips(blob, expect)


def test_round_trip_json_fallback(monkeypatch):
    # Force the dependency-free path even when msgpack is installed.
    monkeypatch.setattr(pose_codec, "msgpack", None)
    blob = pose_codec.encode_poses(_POSES, score=42.5, method="minimize")
    _assert_round_trips(blob, "json+b64")


def test_msgpack_blob_needs_msgpack(monkeypatch):
    if pose_codec.msgpack is None:
        pytest.skip("msgpack not installed")
    blob = pose_codec.encode_poses(_POSES, score=42.5, method="minimize")
    monkeypatch.setattr(pose_codec, "msgpack", None)
    with pytest.raises(ValueError, match="msgpack"):
        pose_codec.decode_poses(blob)


def test_unknown_codec_raises():
    with pytest.raises(ValueError, match="unknown pose codec"):
        pose_codec.decode_poses({"codec": "bogus", "data": ""})
