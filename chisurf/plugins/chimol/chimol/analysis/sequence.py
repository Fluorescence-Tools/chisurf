from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _coerce_residue_vector(values: Optional[Sequence[object]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        arr = np.asarray(values)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    try:
        out = arr.astype(int)
    except Exception:
        return None
    return out


def build_residue_alignment(
    residue_numbers_map: Mapping[str, Optional[Sequence[object]]],
    lengths: Mapping[str, int],
) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]]]:
    """Build a global residue-number axis for all loaded objects.

    Parameters
    ----------
    residue_numbers_map:
        Mapping from object id to a 1D iterable of PDB residue numbers aligned
        with the CA trace for that object. Values may be ``None``.
    lengths:
        Mapping from object id to the sequence length (number of CA residues).

    Returns
    -------
    axis:
        1D array of residue numbers representing the global alignment axis.
        May be empty when no residue information is available.
    index_maps:
        Dict mapping each object id to a 1D int array of shape ``(len(axis),)``
        giving the sequence index (0-based along that object's CA trace) that
        contributes to each axis position, or ``-1`` where the object has a
        gap / no residue for that number. When residue numbers are not
        available for any object, the axis falls back to a simple
        1..max_len index and index_maps map columns to plain sequence indices.
    """

    # Collect only objects that provide a usable residue-number vector.
    valid: Dict[str, np.ndarray] = {}
    for key, values in residue_numbers_map.items():
        arr = _coerce_residue_vector(values)
        if arr is not None and arr.size > 0:
            valid[str(key)] = arr

    # Fallback: no PDB residue numbers anywhere -> treat sequence index as axis.
    if not valid:
        max_len = max(lengths.values()) if lengths else 0
        if max_len <= 0:
            axis = np.zeros(0, dtype=int)
            index_maps: Dict[str, Optional[np.ndarray]] = {
                str(k): None for k in lengths.keys()
            }
            return axis, index_maps

        axis = np.arange(1, max_len + 1, dtype=int)
        index_maps: Dict[str, Optional[np.ndarray]] = {}
        for key, length in lengths.items():
            n = int(length or 0)
            if n <= 0:
                index_maps[str(key)] = None
                continue
            limit = min(max_len, n)
            mapping = np.full(max_len, -1, dtype=int)
            mapping[:limit] = np.arange(limit, dtype=int)
            index_maps[str(key)] = mapping
        return axis, index_maps

    # Build global numeric residue axis from the minimum to maximum PDB resid.
    global_min = min(int(np.min(arr)) for arr in valid.values())
    global_max = max(int(np.max(arr)) for arr in valid.values())
    if global_max < global_min:
        axis = np.zeros(0, dtype=int)
        index_maps = {str(k): None for k in lengths.keys()}
        return axis, index_maps

    axis = np.arange(global_min, global_max + 1, dtype=int)
    size = int(axis.shape[0])

    index_maps: Dict[str, Optional[np.ndarray]] = {}
    for key, length in lengths.items():
        obj_id = str(key)
        arr = valid.get(obj_id)
        if arr is None or size <= 0:
            index_maps[obj_id] = None
            continue

        seq_len = int(length or 0)
        eff_len = min(seq_len, int(arr.shape[0]))
        mapping = np.full(size, -1, dtype=int)
        for seq_idx in range(eff_len):
            try:
                r = int(arr[seq_idx])
            except Exception:
                continue
            pos = r - global_min
            if 0 <= pos < size and mapping[pos] < 0:
                mapping[pos] = seq_idx
        index_maps[obj_id] = mapping

    return axis, index_maps


def needleman_wunsch(
    seq1: Sequence[object],
    seq2: Sequence[object],
    match_score: float = 1.0,
    mismatch_score: float = -1.0,
    gap_penalty: float = -1.0,
) -> Tuple[Sequence[object], Sequence[object], float]:
    """Global alignment (Needleman–Wunsch).

    This implementation is small, self-contained, and independent from the
    ProtView sequence viewer. It is provided for future analysis tools and
    MUST NOT be used to alter sequence numbering, which is derived solely
    from PDB residue ids.
    """

    s1 = list(seq1 or [])
    s2 = list(seq2 or [])
    n = len(s1)
    m = len(s2)
    if n == 0 and m == 0:
        return [], [], 0.0

    score = np.zeros((n + 1, m + 1), dtype=float)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)

    # Initialize first row/column with gap penalties.
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap_penalty
        trace[i, 0] = 1  # up
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap_penalty
        trace[0, j] = 2  # left

    # Fill DP matrix.
    for i in range(1, n + 1):
        a = s1[i - 1]
        for j in range(1, m + 1):
            b = s2[j - 1]
            if a == b:
                diag = score[i - 1, j - 1] + match_score
            else:
                diag = score[i - 1, j - 1] + mismatch_score
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            best = diag
            move = 0  # diag
            if up > best:
                best = up
                move = 1
            if left > best:
                best = left
                move = 2
            score[i, j] = best
            trace[i, j] = move

    # Traceback from bottom-right.
    aligned1: list[object] = []
    aligned2: list[object] = []
    i = n
    j = m
    while i > 0 or j > 0:
        move = trace[i, j]
        if i > 0 and j > 0 and move == 0:
            aligned1.append(s1[i - 1])
            aligned2.append(s2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):  # up -> gap in seq2
            aligned1.append(s1[i - 1])
            aligned2.append("-")
            i -= 1
        else:  # left -> gap in seq1
            aligned1.append("-")
            aligned2.append(s2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    return aligned1, aligned2, float(score[n, m])


def smith_waterman(
    seq1: Sequence[object],
    seq2: Sequence[object],
    match_score: float = 2.0,
    mismatch_score: float = -1.0,
    gap_penalty: float = -1.0,
) -> Tuple[Sequence[object], Sequence[object], float]:
    """Local alignment (Smith–Waterman).

    As with :func:`needleman_wunsch`, this routine is intended for analysis
    and must not be used to alter residue numbering in the viewer.
    """

    s1 = list(seq1 or [])
    s2 = list(seq2 or [])
    n = len(s1)
    m = len(s2)
    if n == 0 or m == 0:
        return [], [], 0.0

    score = np.zeros((n + 1, m + 1), dtype=float)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)

    max_score = 0.0
    max_pos = (0, 0)

    for i in range(1, n + 1):
        a = s1[i - 1]
        for j in range(1, m + 1):
            b = s2[j - 1]
            if a == b:
                diag = score[i - 1, j - 1] + match_score
            else:
                diag = score[i - 1, j - 1] + mismatch_score
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            best = 0.0
            move = 3  # 3 == stop
            if diag > best:
                best = diag
                move = 0
            if up > best:
                best = up
                move = 1
            if left > best:
                best = left
                move = 2
            score[i, j] = best
            trace[i, j] = move
            if best > max_score:
                max_score = best
                max_pos = (i, j)

    aligned1: list[object] = []
    aligned2: list[object] = []
    i, j = max_pos
    while i > 0 and j > 0:
        move = trace[i, j]
        if move == 3 or score[i, j] <= 0.0:
            break
        if move == 0:
            aligned1.append(s1[i - 1])
            aligned2.append(s2[j - 1])
            i -= 1
            j -= 1
        elif move == 1:
            aligned1.append(s1[i - 1])
            aligned2.append("-")
            i -= 1
        elif move == 2:
            aligned1.append("-")
            aligned2.append(s2[j - 1])
            j -= 1
        else:
            break

    aligned1.reverse()
    aligned2.reverse()
    return aligned1, aligned2, float(max_score)
