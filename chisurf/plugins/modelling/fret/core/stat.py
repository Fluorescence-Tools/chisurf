"""Read PMI ``stat.*.out`` files for live docking progress and score curves.

The PMI ``ReplicaExchange`` macro appends one line per frame to ``stat.0.out``:
a first header line (string keys describing each column) followed by one
``repr(dict)`` per frame whose integer keys map to the header columns. We only
need two columns — the frame index and the total score — to drive a progress
bar and a score-vs-frame convergence plot, so the parsing here is deliberately
tolerant: unreadable lines (including the giant header) are skipped.
"""

from __future__ import annotations

import ast
import os
from typing import List, Tuple

#: Integer column keys PMI uses in the per-frame stat dicts.
_TOTAL_SCORE_KEY = 1
_NFRAME_KEY = 4

# Field names in the header line, in case PMI ever renumbers the columns.
_TOTAL_SCORE_NAME = "Total_Score"
_NFRAME_NAME = "MonteCarlo_Nframe"


def count_frames(stat_path) -> int:
    """Return the number of completed frames recorded in ``stat_path``.

    One header line plus one line per frame, so ``data_lines = total - 1``.
    Returns 0 when the file is missing or unreadable.
    """
    try:
        with open(stat_path, "r", encoding="utf-8", errors="ignore") as fh:
            n = sum(1 for _ in fh)
    except OSError:
        return 0
    return max(0, n - 1)


def _resolve_keys(header_line: str) -> Tuple[int, int]:
    """Map the score/frame columns from the header line; fall back to defaults."""
    score_key, frame_key = _TOTAL_SCORE_KEY, _NFRAME_KEY
    for col, name in _safe_dict(header_line).items():
        if name == _TOTAL_SCORE_NAME and isinstance(col, int):
            score_key = col
        elif name == _NFRAME_NAME and isinstance(col, int):
            frame_key = col
    return score_key, frame_key


def _safe_dict(line: str) -> dict:
    """``ast.literal_eval`` a stat line into a dict, or ``{}`` if it cannot."""
    line = line.strip()
    if not line.startswith("{"):
        return {}
    try:
        obj = ast.literal_eval(line)
    except (ValueError, SyntaxError):  # e.g. header's environ(...) call
        return {}
    return obj if isinstance(obj, dict) else {}


def read_score_series(stat_path) -> Tuple[List[float], List[float]]:
    """Return ``(frames, scores)`` parsed from a PMI stat file.

    Parameters
    ----------
    stat_path : str
        Path to a ``stat.0.out`` file (need not be complete; partial files from
        a running job are fine).

    Returns
    -------
    (list of float, list of float)
        Frame indices and the matching total scores, in file order. Empty when
        the file is missing or has no parseable data lines.
    """
    if not os.path.exists(stat_path):
        return [], []
    frames: List[float] = []
    scores: List[float] = []
    score_key, frame_key = _TOTAL_SCORE_KEY, _NFRAME_KEY
    with open(stat_path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line.startswith("{"):  # PMI Monte-Carlo stat dict
                if i == 0:
                    score_key, frame_key = _resolve_keys(line)
                    continue
                d = _safe_dict(line)
                if score_key not in d:
                    continue
                try:
                    scores.append(float(d[score_key]))
                    frames.append(float(d.get(frame_key, len(frames))))
                except (TypeError, ValueError):
                    continue
            else:  # plain "frame,score" convergence CSV (minimisation)
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    frames.append(float(parts[0]))
                    scores.append(float(parts[1]))
                except ValueError:
                    continue  # header row "frame,score"
    return frames, scores


__all__ = ["count_frames", "read_score_series"]
