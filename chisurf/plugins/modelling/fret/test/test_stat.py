"""Tests for the PMI stat-file parser (progress + score curves)."""

from __future__ import annotations

from chisurf.plugins.modelling.fret.core import stat

_HEADER = (
    "{'STAT2HEADER': 'STAT2HEADER', 0: 'AV_Score', 1: 'Total_Score', "
    "4: 'MonteCarlo_Nframe'}\n"
)


def test_count_frames(tmp_path):
    p = tmp_path / "stat.0.out"
    p.write_text(_HEADER + "{1: '10.0', 4: '0'}\n{1: '9.0', 4: '1'}\n")
    assert stat.count_frames(p) == 2
    assert stat.count_frames(tmp_path / "missing.out") == 0


def test_read_score_series(tmp_path):
    p = tmp_path / "stat.0.out"
    p.write_text(_HEADER + "{1: '10.0', 4: '0'}\n{1: '7.5', 4: '1'}\n{1: '5.0', 4: '2'}\n")
    frames, scores = stat.read_score_series(p)
    assert frames == [0.0, 1.0, 2.0]
    assert scores == [10.0, 7.5, 5.0]


def test_read_score_series_skips_unparseable_header(tmp_path):
    # a header containing a call like environ(...) must not break parsing
    p = tmp_path / "stat.0.out"
    p.write_text(
        "{'STAT2HEADER_ENVIRON': environ({'X': '1'}), 1: 'Total_Score', 4: 'MonteCarlo_Nframe'}\n"
        "{1: '3.0', 4: '0'}\n"
    )
    frames, scores = stat.read_score_series(p)
    assert scores == [3.0]


def test_read_score_series_missing_file(tmp_path):
    assert stat.read_score_series(tmp_path / "nope.out") == ([], [])


def test_read_convergence_csv(tmp_path):
    # the minimisation engine writes a plain "frame,score" CSV
    p = tmp_path / "convergence.csv"
    p.write_text("frame,score\n0,100.0\n1,40.0\n2,33.0\n")
    frames, scores = stat.read_score_series(p)
    assert frames == [0.0, 1.0, 2.0]
    assert scores == [100.0, 40.0, 33.0]
    assert stat.count_frames(p) == 3
