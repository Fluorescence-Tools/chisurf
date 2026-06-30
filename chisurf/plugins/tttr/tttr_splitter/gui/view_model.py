"""Qt-free view-model backing the TTTR Split / Convert tool.

:class:`SplitterViewModel` holds the interactive state (loaded TTTR object, the
split/convert options that AutoForm binds its controls to, and the batch file
list) and performs the split/transcode through :mod:`tttrlib`. It is deliberately
free of Qt so the logic can be unit-tested headlessly; the GUI (``gui.tool`` +
``gui.sections``) owns all Qt concerns and drives this model.

Mirrors :class:`chisurf.plugins.microscopy.psf_determination.gui.view_model.PsfViewModel`.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

import tttrlib

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "splitter.view.json"

# Unified mapping: container name → (file-extension stem, record-type id, container id)
_CONTAINER_INFO = {
    "PTU": ("ptu", 4, 0),  # PQ_PTU_CONTAINER → PQ_RECORD_TYPE_HHT3v2
    "HT3": ("ht3", 4, 1),  # PQ_HT3_CONTAINER → PQ_RECORD_TYPE_HHT3v2
    "SPC-130": ("spc", 7, 2),  # BH_SPC130_CONTAINER → BH_RECORD_TYPE_SPC130
    "SPC-600_256": ("spc", 8, 3),  # BH_SPC600_256_CONTAINER → BH_RECORD_TYPE_SPC600_256
    "SPC-600_4096": ("spc", 9, 4),  # BH_SPC600_4096_CONTAINER → BH_RECORD_TYPE_SPC600_4096
    "PHOTON-HDF5": ("hdf", 4, 5),  # PHOTON_HDF_CONTAINER → PQ_RECORD_TYPE_PHT3
    "CZ-RAW": ("raw", 10, 6),  # CZ_CONFOCOR3_CONTAINER → CZ_RECORD_TYPE_CONFOCOR3
    "SM": ("sm", 11, 7),  # SM_CONTAINER → SM_RECORD_TYPE
}

_SPC_TYPES = {"SPC-130", "SPC-600_256", "SPC-600_4096"}


class SplitterViewModel:
    """State + logic for the TTTR Split / Convert tool (no Qt)."""

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``splitter.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        # ── AutoForm-bound settings ─────────────────────────────────────
        self.input_format = "Auto"
        self.output_format = "PTU"
        self.photons_per_file_k = 300
        self.microtime_binning = "1"
        self.split_files = True
        self.reset_macro_times = True
        self.keep_original = True
        self.batch_use_parent = True

        # ── runtime state (driven by the custom GUI sections) ───────────
        self.input_file = ""
        self.output_folder = ""
        self.batch_files: list[str] = []
        self._tttr = None
        self._observers: list[Callable[[str], None]] = []

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("splitter observer failed", exc_info=True)

    def update(self) -> None:
        """Refresh observers after a bound field changes (AutoForm hook)."""
        self.notify("fields")

    # ── AutoForm options sources ───────────────────────────────────────
    def input_format_options(self) -> list[str]:
        """Input container choices: ``Auto`` plus every tttrlib container name."""
        return ["Auto", *tttrlib.TTTR.get_supported_container_names()]

    def output_format_options(self) -> list[str]:
        """Output container choices (every supported tttrlib container name)."""
        return list(tttrlib.TTTR.get_supported_container_names())

    # ── loaded-file accessors ──────────────────────────────────────────
    def set_tttr(self, tttr, path: str) -> None:
        """Store a freshly loaded TTTR object and its source path."""
        self._tttr = tttr
        self.input_file = str(path)
        if not self.output_folder:
            self.output_folder = str(pathlib.Path(path).parent)
        self.notify("loaded")

    @property
    def tttr(self):
        """The currently loaded ``tttrlib.TTTR`` object, or ``None``."""
        return self._tttr

    @property
    def tttr_type(self) -> str | None:
        """The forced input container, or ``None`` when ``Auto`` is selected."""
        return None if self.input_format == "Auto" else self.input_format

    @property
    def input_tttr_type_id(self) -> int:
        """Numeric tttrlib container id of the input (inferred when ``Auto``)."""
        if self.tttr_type is None:
            return tttrlib.inferTTTRFileType(self.tttr_input_filename.as_posix())
        names = tttrlib.TTTR.get_supported_container_names()
        return names.index(self.input_format)

    @property
    def tttr_input_filename(self) -> pathlib.Path | None:
        """The input path as a ``Path`` if it points at an existing file."""
        path = pathlib.Path(self.input_file.strip())
        return path if path.is_file() else None

    @property
    def output_folder_path(self) -> pathlib.Path | None:
        """Output folder as a ``Path``, creating it on demand; ``None`` if invalid."""
        folder = self.output_folder.strip()
        if not folder:
            return None
        p = pathlib.Path(folder)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Failed to create folder '%s': %s", p, exc)
                return None
        return p

    @property
    def photons_per_file(self) -> int:
        """Chunk size in photons (the ``×1000`` spin value)."""
        return int(self.photons_per_file_k) * 1000

    @property
    def micro_time_binning(self) -> int:
        """Micro-time binning factor (clamped to ≥8 for SPC containers)."""
        default_bin = 1
        try:
            names = tttrlib.TTTR.get_supported_container_names()
            container = names[self._tttr.header.tttr_container_type]
            if container in _SPC_TYPES:
                default_bin = 8
        except Exception:
            if self.tttr_type in _SPC_TYPES:
                default_bin = 8
        try:
            user_bin = int(self.microtime_binning)
        except (ValueError, TypeError):
            user_bin = default_bin
        return max(default_bin, user_bin)

    # ── core operation ─────────────────────────────────────────────────
    def can_split(self) -> str | None:
        """Return ``None`` when a split can run, else a human-readable reason."""
        if self._tttr is None:
            return "Please load a TTTR file first."
        if len(self._tttr) == 0:
            return "No photons to split!"
        if self.output_folder_path is None:
            return "Please specify a valid output folder."
        return None

    def do_split(self, progress_cb: Callable[[int], None] | None = None) -> pathlib.Path:
        """Split / transcode the loaded TTTR file into the output folder.

        Writes either one file per ``photons_per_file`` chunk (``split_files``)
        or a single file with all photons, converting the container format when
        the output differs from the input. ``progress_cb`` (if given) receives a
        0–100 percentage as each chunk is written. Returns the output sub-folder.
        """
        reason = self.can_split()
        if reason is not None:
            raise ValueError(reason)

        t = self._tttr
        total = len(t)
        out_folder = self.output_folder_path

        # Build the photon-index ranges.
        if self.split_files:
            chunk = self.photons_per_file
            full, rem = divmod(total, chunk)
            n = full + (1 if rem else 0)
            ranges = [(i * chunk, min((i + 1) * chunk, total)) for i in range(n)]
        else:
            ranges = [(0, total)]
            n = 1

        in_path = self.tttr_input_filename
        subfolder = out_folder / in_path.stem
        subfolder.mkdir(parents=True, exist_ok=True)

        # Prepare the (possibly transcoded) header.
        header = t.header
        out_name = self.output_format
        ext, rec, cont = _CONTAINER_INFO[out_name]
        tttr_type_id_output = _CONTAINER_INFO[out_name][2]
        tttr_type_id_input = self.input_tttr_type_id

        if tttr_type_id_input != tttr_type_id_output:
            logger.info("Transcoding %s → %s", in_path, out_name)
            header.tttr_container_type = cont
            header.tttr_record_type = rec
            if out_name == "PTU":
                # PTU via HydraHarp wants the special tag group 0x00010304.
                header.set_tag("TTResultFormat_TTTRRecType", 0x00010304, 268435464)
                header.set_tag("TTResultFormat_BitsPerRecord", 32, 268435464)
                header.set_tag("MeasDesc_RecordType", rec, 268435464)
            else:
                # 268435464 == 0x10000008 (Int8) — the tag value type tttrlib expects.
                header.set_tag("TTResultFormat_TTTRRecType", rec, 268435464)
                header.set_tag("MeasDesc_RecordType", rec, 268435464)
        else:
            logger.info("Splitting %s", in_path)

        # Write each range.
        for i, (start, stop) in enumerate(ranges):
            if progress_cb is not None:
                progress_cb(int((i / n) * 100))

            chunk_tttr = t[start:stop]
            if self.reset_macro_times or self.tttr_type != out_name:
                mt, ut = chunk_tttr.macro_times, chunk_tttr.micro_times
                rc, et = chunk_tttr.routing_channels, chunk_tttr.event_types
                mt0 = -int(mt[0]) if (self.reset_macro_times and len(mt)) else 0
                new_tttr = tttrlib.TTTR()
                new_tttr.append_events(mt, ut, rc, et, True, mt0)
                chunk_tttr = new_tttr

            if self.split_files:
                fname = f"{in_path.stem}_{i:05d}.{ext}"
            else:
                fname = f"{in_path.stem}_all.{ext}"
            chunk_tttr.write(str(subfolder / fname), header)

        if progress_cb is not None:
            progress_cb(100)

        if not self.keep_original:
            in_path.unlink()

        return subfolder

    # ── batch ──────────────────────────────────────────────────────────
    def run_batch(
        self,
        file_progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Split every file in :attr:`batch_files`.

        For each file the loaded TTTR is replaced and :meth:`do_split` is run with
        the current option set; when :attr:`batch_use_parent` is True the output
        folder follows each file's own parent directory. ``file_progress_cb`` (if
        given) is called as ``(index, count, path)`` before each file. Returns the
        number of files processed.
        """
        files = list(self.batch_files)
        count = len(files)
        keep_folder = self.output_folder
        for i, path_str in enumerate(files):
            if file_progress_cb is not None:
                file_progress_cb(i, count, path_str)
            try:
                tttr_type = self.tttr_type
                tttr = (
                    tttrlib.TTTR(path_str)
                    if tttr_type is None
                    else tttrlib.TTTR(path_str, tttr_type)
                )
                self.set_tttr(tttr, path_str)
                if self.batch_use_parent:
                    self.output_folder = str(pathlib.Path(path_str).parent)
                else:
                    self.output_folder = keep_folder
                self.do_split()
            except Exception:
                logger.exception("Batch: failed processing %s", path_str)
                continue
        self.output_folder = keep_folder
        return count


__all__ = ["SplitterViewModel"]
