from __future__ import annotations
from chisurf import typing

import csv
import os
import re
import numpy as np

import chisurf.fio as io
import chisurf
from chisurf import logging


# ----------------------------- Utilities ---------------------------------

def _open_maybe_zipped(filename: str, mode: str = "r"):
    """
    Compatibility wrapper for zipped/plain I/O across chisurf versions.
    """
    try:
        opener = io.open_maybe_zipped
        used = "io.open_maybe_zipped"
    except AttributeError:
        opener = io.zipped.open_maybe_zipped
        used = "io.zipped.open_maybe_zipped"
    logging.debug(f"_open_maybe_zipped: using {used} with mode={mode} for filename={filename}")
    return opener(filename=filename, mode=mode)


def _decode(line) -> str:
    if isinstance(line, bytes):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return line.decode(enc, errors="ignore")
            except Exception:
                pass
        return line.decode("utf-8", errors="ignore")
    return str(line)


def _tokenize(line: str, delimiter_hint: str | None) -> list[str]:
    if delimiter_hint is None:
        return line.strip().split()
    return [t for t in line.strip().split(delimiter_hint)]


### NEW: helper to test numeric tokens (supports decimal comma if requested)
def _is_number_token(tok: str, decimal_comma: bool = False) -> bool:
    s = _decode(tok).strip()
    if not s:
        return False
    if decimal_comma:
        s = s.replace(",", ".")
    try:
        float(s)
        return True
    except Exception:
        return False


def _find_table_region(fp, max_scan_lines: int = 5000) -> tuple[int, int | None, str | None, int, bool, list[str]]:
    """
    Pure column-count based detection (no block markers, no numeric heuristics), with a
    small addition: if the first complete-width row contains non-numeric tokens,
    it is treated as a header row and will be skipped.

    Returns:
      skiprows (index of first *data* row),
      header_idx (index of header line or None),
      delimiter (str|None),
      ncols (complete width),
      dec_comma (bool),
      sniff_sample (list[str]) preview.
    """
    # Read lines
    lines: list[str] = []
    for _ in range(max_scan_lines):
        ln = fp.readline()
        if not ln:
            break
        lines.append(_decode(ln))

    N = len(lines)
    candidates = [",", "\t", ";", "|", None]  # None => whitespace
    has_comma_digits = any(re.search(r"\d,\d", ln) for ln in lines[:min(N, 2000)])

    best = dict(score=-1.0, delim=None, complete_cols=0, first_idx=0, header_idx=None, dec_comma=False)

    for delim in candidates:
        # Build ncols per line (0 for empty / single token -> not a table row)
        ncols = []
        for i in range(N):
            s = lines[i].rstrip("\n")
            if not s.strip():
                ncols.append(0)
                continue
            toks = _tokenize(s, delim)
            ncols.append(len(toks) if len(toks) >= 2 else 0)

        # Frequency of column counts ≥ 2
        counts = {}
        for c in ncols:
            if c >= 2:
                counts[c] = counts.get(c, 0) + 1
        if not counts:
            continue

        # Complete width: most frequent; tie -> larger width
        complete_cols = max(sorted(counts.keys()), key=lambda c: (counts[c], c))

        # First complete row index
        try:
            first_complete = next(i for i, c in enumerate(ncols) if c == complete_cols)
        except StopIteration:
            continue

        # Decimal-comma possibility for this delimiter
        dec_flag = bool(has_comma_digits and (delim in (";", "\t", "|", None)))

        ### NEW: decide whether the first complete-width row is a header row
        toks_first = _tokenize(lines[first_complete], delim)
        non_numeric_in_first = any(not _is_number_token(t, decimal_comma=dec_flag) for t in toks_first)
        # By default, assume header is the line *before* the first complete row (legacy behavior)
        header_idx = None
        if first_complete > 0:
            prev = first_complete - 1
            prev_cols = ncols[prev]
            enough_cols_threshold = max(2, complete_cols // 2)
            if prev_cols == 0:
                header_idx = prev  # unsplittable -> header/preamble
            elif 2 <= prev_cols < complete_cols and prev_cols >= enough_cols_threshold:
                header_idx = prev   # splits but fewer cols -> header
            # If blank just before, allow stepping back one more if that line splits "enough"
            if header_idx is None and prev_cols == 0 and prev - 1 >= 0 and ncols[prev - 1] >= enough_cols_threshold:
                header_idx = prev - 1

        # If the first complete-width row looks like a header (contains non-numeric tokens),
        # promote it to the header and advance the first data row by one.
        first_idx_use = first_complete
        if non_numeric_in_first:
            header_idx = first_complete
            first_idx_use = first_complete + 1  # skip header row in data

        # Score this delimiter: length of consecutive complete-width block from first_complete + richness
        run_len = 0
        j = first_complete
        while j < N and ncols[j] == complete_cols:
            run_len += 1
            j += 1
        richness = float(np.log(max(complete_cols, 2)))
        score = run_len * richness

        if score > best["score"]:
            best.update(
                score=score,
                delim=delim,
                complete_cols=complete_cols,
                first_idx=first_idx_use,  # NOTE: uses possibly +1 if header promoted
                header_idx=header_idx,
                dec_comma=dec_flag
            )

    # Fallback if nothing usable was found
    if best["score"] < 0:
        sniff_sample = lines[: min(15, N)]
        logging.debug("_find_table_region: no usable table detected; returning defaults")
        return 0, None, None, 0, False, sniff_sample

    # Build small preview around the detected start
    sample_start = max(0, best["first_idx"] - 3)
    sample_end = min(N, best["first_idx"] + 12)
    sniff_sample = [ln for ln in lines[sample_start:sample_end] if ln.strip()]

    logging.debug(
        f"_find_table_region: best delim={repr(best['delim'])} complete_cols={best['complete_cols']} "
        f"first_idx={best['first_idx']} header_idx={best['header_idx']} decimal_comma={best['dec_comma']}"
    )
    return best["first_idx"], best["header_idx"], best["delim"], best["complete_cols"], best["dec_comma"], sniff_sample


def _decimal_comma_converters(ncols: int):
    """
    Build converters dict for genfromtxt to interpret decimal commas (e.g., 1,23) as floats.
    """
    def conv_factory():
        return lambda s: float(_decode(s).strip().replace(",", "."))
    convs = {i: conv_factory() for i in range(ncols)}
    logging.debug(f"_decimal_comma_converters: created converters for ncols={ncols}")
    return convs


# ----------------------------- Public API ---------------------------------


def save_xy(
        filename: str,
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool = chisurf.settings.cs_settings['verbose'],
        fmt: str = "%.3f\t%.3f",
        header_string: str = None
) -> None:
    """
    Saves data x, y to file in format (csv/tsv). x and y should have the same length.
    """
    if verbose:
        logging.info("Writing histogram to file: %s" % filename)
        try:
            logging.debug(f"save_xy: n_points={min(len(x), len(y))} fmt={fmt!r} header_present={header_string is not None}")
        except Exception:
            pass
    with _open_maybe_zipped(filename=filename, mode='w') as fp:
        if header_string is not None:
            fp.write(header_string)
            if not header_string.endswith("\n"):
                fp.write("\n")
        for a, b in zip(x, y):
            fp.write(fmt % (a, b))
            fp.write("\n")


def load_xy(
        filename: str,
        verbose: bool = chisurf.settings.cs_settings['verbose'],
        usecols: typing.Tuple[int, int] = None,
        skiprows: int = 0,
        delimiter: str = "\t"
) -> typing.Tuple[np.array, np.array]:
    if usecols is None:
        usecols = [0, 1]
    if verbose:
        logging.info(f"Loading file: {filename}")
        logging.debug(f"load_xy params: usecols={usecols}, skiprows={skiprows}, delimiter={repr(delimiter)}")
    data = np.loadtxt(
        filename,
        skiprows=skiprows,
        usecols=usecols,
        delimiter=delimiter
    )
    logging.debug(f"load_xy loaded array shape={getattr(data, 'shape', None)} dtype={getattr(data, 'dtype', None)}")
    return data.T[0], data.T[1]


class Csv(object):
    """
    Csv is a class to handle delimited text files (CSV/TSV/whitespace) with robust, marker-free
    detection of header and data regions based purely on column counts.

    Examples
    --------
    Two-column data

    >>> import chisurf.fio.ascii
    >>> csv = chisurf.fio.ascii.Csv()
    >>> filename = './test/data/tcspc/ibh_sample/Decay_577D.txt'
    >>> csv.load(filename)
    >>> csv.data
    array([...])

    One-column Jordi data

    >>> csv = chisurf.fio.ascii.Csv()
    >>> filename = './test/data/tcspc/ibh_sample/Decay_577D.txt'
    >>> csv.load(filename)
    >>> csv.data_x
    array([...])
    >>> csv.data_y
    array([...])
    """

    def __init__(
            self,
            *args,
            filename: str = None,
            colspecs: typing.List[int] = None,
            use_header: bool = False,
            x_on: bool = True,
            y_on: bool = True,
            col_x: int = 0,
            col_y: int = 1,
            col_ex: int = 2,
            col_ey: int = 3,
            reverse: bool = False,
            error_x_on: bool = False,
            directory: str = '.',
            skiprows: int = 9,
            verbose: bool = chisurf.settings.cs_settings['verbose'],
            file_type: str = 'csv',
            **kwargs
    ):
        self._filename = filename
        self.use_header = use_header
        self.x_on = x_on
        self.error_y_on = y_on
        self.col_x = col_x
        self.col_y = col_y
        self.col_ex = col_ex
        self.col_ey = col_ey
        self.reverse = reverse
        self.error_x_on = error_x_on
        self.directory = directory
        # keep attribute for compatibility; not automatically applied anymore
        self.skiprows = skiprows
        self.file_type = file_type
        self.verbose = verbose

        self._header: list[str] | None = None
        self._data = kwargs.get('data', None)

        if colspecs is None:
            colspecs = (15, 17, 17)
        self.colspecs = colspecs

        if self.verbose:
            try:
                logging.debug(
                    "Csv.__init__: "
                    f"use_header={self.use_header} x_on={self.x_on} y_on={self.error_y_on} "
                    f"col_x={self.col_x} col_y={self.col_y} reverse={self.reverse} "
                    f"file_type={self.file_type} skiprows={self.skiprows} directory={self.directory!r} "
                    f"colspecs={self.colspecs}"
                )
            except Exception:
                pass

        if isinstance(filename, str):
            self.load(filename)

    @property
    def filename(self) -> str:
        """The currently open filename."""
        return self._filename

    def load(
            self,
            filename: str,
            skiprows: int = None,
            use_header: bool = None,
            verbose: bool = chisurf.settings.cs_settings['verbose'],
            delimiter: str = None,
            file_type: str = None,
            infer_delimiter: bool = True,
            usecols: typing.List[int] = None,
            **kwargs
    ) -> None:
        """
        Load a file into the Csv object.

        Heuristics:
          - Reads a chunk of the file and determines the *complete* column width (most frequent width ≥ 2).
          - The first row with this width is treated as the first data row (skiprows).
          - If that first complete-width row contains non-numeric tokens (e.g., "Chan\tData"),
            it is treated as a header row and skipped by default.
          - The line immediately before it is treated as header if it splits and has "enough" columns,
            or if it doesn't split at all (considered a header/preamble line).
          - Delimiter is inferred among {',', '\\t', ';', '|', whitespace}.
          - Decimal commas are handled by providing converters to np.genfromtxt.
        """
        if filename is None:
            return None
        if file_type is None:
            file_type = self.file_type
        if use_header is None:
            use_header = self.use_header

        if os.path.isfile(filename):
            self.directory = os.path.dirname(filename)
            self._filename = filename
            colspecs = self.colspecs

            # ---- Auto-detect table region (column-count logic + header-on-first-row handling) ----
            try:
                with _open_maybe_zipped(filename=filename, mode='r') as fp:
                    auto_skip, header_idx, auto_delim, ncols_auto, dec_comma, sniff_sample = _find_table_region(fp)
            except Exception:
                auto_skip, header_idx, auto_delim, ncols_auto, dec_comma, sniff_sample = 0, None, None, 0, False, []

            # Decide skiprows
            if skiprows is None:
                skiprows_eff = auto_skip  # already advanced past header row if it was detected there
            else:
                # If user provided a value, respect it (but don't undercut auto_skip if they passed too small).
                skiprows_eff = max(skiprows, auto_skip)

            # Decide delimiter
            if delimiter is None and infer_delimiter:
                delimiter = auto_delim  # may be None -> whitespace

            # Capture header tokens if we detected a header (for Csv.header)
            self._header = None
            if header_idx is not None:
                try:
                    with _open_maybe_zipped(filename=filename, mode='r') as fp:
                        for _ in range(header_idx):
                            fp.readline()
                        hdr_line = _decode(fp.readline())
                    toks = _tokenize(hdr_line, delimiter)
                    toks = [t.strip() for t in toks if t.strip() != ""]
                    if toks:
                        self._header = toks
                        if self.verbose:
                            logging.debug(f"Csv.load: header tokens detected n={len(toks)} sample={toks[:min(5,len(toks))]}")
                except Exception:
                    self._header = None

            if self.verbose:
                logging.info(f"Reading: {filename}")
                logging.info(f"Auto-detected skiprows: {auto_skip} → Using: {skiprows_eff}")
                logging.info(f"Detected delimiter: {repr(delimiter)}")
                logging.info(f"Decimal comma detected: {bool(dec_comma)}")
                logging.info(f"Detected header: {bool(self._header)}")
                try:
                    preview = "".join(sniff_sample)[:512]
                    logging.info(preview)
                except Exception:
                    pass

            # ---- Load data ----
            if file_type == 'csv':
                load_kwargs = dict(
                    fname=filename,
                    delimiter=delimiter,                  # None → any whitespace
                    skip_header=skiprows_eff if skiprows_eff is not None else 0,
                    usecols=usecols,
                    autostrip=True,
                    comments=None,
                    invalid_raise=False,
                    filling_values=np.nan
                )
                if dec_comma and delimiter != ",":
                    max_cols = ncols_auto if ncols_auto > 0 else 256
                    load_kwargs["converters"] = _decimal_comma_converters(max_cols)

                if self.verbose:
                    # Log a compact view of kwargs without dumping converters functions
                    kw_preview = {k: ("<converters>" if k == "converters" else v) for k, v in load_kwargs.items()}
                    logging.debug(f"Csv.load: np.genfromtxt kwargs: {kw_preview}")
                d = np.genfromtxt(**load_kwargs)

            else:
                # fixed-width fallback (rare)
                d = np.genfromtxt(
                    fname=filename,
                    delimiter=colspecs,
                    skip_header=skiprows_eff if skiprows_eff is not None else 0,
                    usecols=usecols,
                    names='infer' if use_header else None,
                    comments=None,
                    invalid_raise=False,
                    filling_values=np.nan,
                    **kwargs
                )

            # Normalize shape for downstream code
            if d is None or (isinstance(d, float) or isinstance(d, np.floating)):
                d = np.array([[d]])
            elif d.ndim == 1:
                d = d.reshape(-1, 1)

            if self.verbose:
                try:
                    logging.info(f"Csv.load: loaded shape={d.shape} dtype={getattr(d, 'dtype', None)}")
                except Exception:
                    pass

            self._data = d
        else:
            chisurf.logging.warning(f"File {filename} not found")

    def save(
            self,
            data: np.ndarray,
            filename: str,
            delimiter: str = '\t',
            file_type: str = 'txt',
            header: str = ''
    ):
        self._data = data
        if self.verbose:
            shape = getattr(data, 'shape', None)
            s = """Saving
            ------
            filename: %s
            reading_routine: %s
            delimiter: %s
            object_type: %s
            shape: %s
            """ % (filename, file_type, delimiter, type(data), shape)
            logging.info(s)
            try:
                hdr_len = len(header.splitlines()) if isinstance(header, str) and header else 0
                logging.debug(f"Csv.save: header lines={hdr_len}")
            except Exception:
                pass
        if file_type == 'txt':
            np.savetxt(
                filename,
                data.T,
                delimiter=delimiter,
                header=header
            )
        if file_type == 'npy':
            np.save(
                filename,
                data.T
            )

    @property
    def n_cols(self) -> int:
        """The number of columns."""
        return self._data.shape[1]

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return self._data.shape[0]

    @property
    def data(self) -> np.array:
        """Numpy array of the data (transposed to match historical behavior)."""
        arr = np.array(self._data, dtype=np.float64).T
        return arr[::-1] if self.reverse else arr

    @property
    def header(self) -> typing.List[str]:
        """
        A list of the column headers; if none detected, return index strings.
        """
        if self._header is not None:
            return [str(i) for i in self._header]
        # fall back to indices based on actual data width
        return [str(i) for i in range(self.data.shape[1])]

    @property
    def n_points(self) -> int:
        """
        The number of data points corresponds to the number of rows.
        """
        return self.n_rows
