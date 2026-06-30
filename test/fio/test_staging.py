"""Tests for the Qt-free slow-storage staging helper.

Covers :mod:`chisurf.core.fio.staging`: the slow/fast decision, progress
reporting, cancellation, ephemeral cleanup, and that a staged load produces
data identical to a direct ``tttrlib`` read.
"""

import pathlib

import pytest

from chisurf.core.fio import staging

HERE = pathlib.Path(__file__).resolve().parents[1]  # test/
PTU = HERE / "data" / "clsm" / "Leica_SP8.ptu"


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the staging cache at a tmp dir so leftovers are easy to assert."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(staging, "_cache_dir", lambda: cache)
    return cache


def _make_file(path: pathlib.Path, size: int) -> pathlib.Path:
    path.write_bytes(bytes((i * 37) % 256 for i in range(size)))
    return path


def test_fast_source_not_staged(tmp_path, isolated_cache):
    src = _make_file(tmp_path / "fast.dat", 4 * 1024 * 1024)
    # Local disk easily beats a 0 MB/s threshold -> never staged.
    local, was_staged = staging.stage_path_if_slow(src, threshold_mbps=0.0, min_size=0)
    assert was_staged is False
    assert local == src
    assert not any(isolated_cache.iterdir())  # no temp created


def test_min_size_gate(tmp_path, isolated_cache):
    src = _make_file(tmp_path / "small.dat", 1024 * 1024)
    # Even a force-slow threshold is skipped when below the min-size gate.
    local, was_staged = staging.stage_path_if_slow(
        src, threshold_mbps=1e9, min_size=8 * 1024 * 1024
    )
    assert was_staged is False
    assert local == src
    assert not any(isolated_cache.iterdir())


def test_slow_source_staged_with_progress(tmp_path, isolated_cache):
    size = 5 * 1024 * 1024
    src = _make_file(tmp_path / "slow.dat", size)

    events = []
    local, was_staged = staging.stage_path_if_slow(
        src,
        threshold_mbps=1e9,  # everything looks slow
        min_size=0,
        probe_bytes=512 * 1024,
        chunk_bytes=256 * 1024,
        progress_cb=lambda *a: events.append(a),
    )

    assert was_staged is True
    assert local.exists()
    assert local.name == src.name  # name preserved for tttrlib
    assert local.read_bytes() == src.read_bytes()  # byte-exact copy
    assert local.parent.parent == isolated_cache  # staged under cache dir

    # progress: monotonic bytes, ends at total, positive speed throughout
    assert len(events) >= 2
    done = [e[0] for e in events]
    assert done == sorted(done)
    assert done[-1] == size
    assert all(e[1] == size for e in events)  # total reported
    assert all(e[2] > 0 for e in events)  # mbps > 0


def test_cancel_mid_copy(tmp_path, isolated_cache):
    src = _make_file(tmp_path / "cancel.dat", 5 * 1024 * 1024)

    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2  # allow the probe, abort during the copy

    with pytest.raises(staging.StagingCancelled):
        staging.stage_path_if_slow(
            src,
            threshold_mbps=1e9,
            min_size=0,
            probe_bytes=256 * 1024,
            chunk_bytes=256 * 1024,
            cancel_cb=cancel,
        )

    # the partial copy must have been removed
    assert not any(isolated_cache.iterdir())


def test_staged_source_cleans_up_even_on_error(tmp_path, isolated_cache):
    src = _make_file(tmp_path / "ctx.dat", 5 * 1024 * 1024)

    captured = {}
    with pytest.raises(ValueError):
        with staging.staged_source(
            src, threshold_mbps=1e9, min_size=0, probe_bytes=256 * 1024
        ) as local:
            captured["local"] = local
            assert local.exists()
            raise ValueError("boom")

    assert not captured["local"].exists()
    assert not any(isolated_cache.iterdir())


def test_disabled_via_settings(tmp_path, isolated_cache, monkeypatch):
    import chisurf.settings as settings

    monkeypatch.setitem(settings.cs_settings, "data_loading", {"enabled": False})
    src = _make_file(tmp_path / "disabled.dat", 5 * 1024 * 1024)
    local, was_staged = staging.stage_path_if_slow(src, threshold_mbps=1e9, min_size=0)
    assert was_staged is False
    assert local == src


@pytest.mark.skipif(not PTU.is_file(), reason="sample PTU not available")
def test_open_tttr_staged_matches_direct():
    tttrlib = pytest.importorskip("tttrlib")
    direct = tttrlib.TTTR(str(PTU)).get_n_valid_events()
    staged = staging.open_tttr(str(PTU), threshold_mbps=1e9, min_size=0)
    assert staged.get_n_valid_events() == direct


@pytest.mark.skipif(not PTU.is_file(), reason="sample PTU not available")
def test_open_tttr_fast_path_no_leftover(isolated_cache):
    pytest.importorskip("tttrlib")
    # Local sample is fast -> not staged, nothing left in the cache.
    staging.open_tttr(str(PTU))
    assert not any(isolated_cache.iterdir())
