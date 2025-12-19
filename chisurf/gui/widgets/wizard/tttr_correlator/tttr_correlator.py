import os
import pathlib
import typing

import tttrlib
import json
import numpy as np

import pyqtgraph as pg

import chisurf
import chisurf.fio as io
import chisurf.gui.decorators
import chisurf.settings
from chisurf.gui import QtGui, QtWidgets, QtCore, uic
from chisurf.fluorescence.fcs.channel_setups import load_fcs_channel_setups
from .tttr_correlator_ui import setup_ui as _setup_ui

colors = chisurf.settings.gui['plot']['colors']


class WizardTTTRCorrelator(QtWidgets.QWizardPage):

    @property
    def analysis_folder(self) -> pathlib.Path:
        return pathlib.Path(str(self.lineEdit_3.text()))

    @property
    def output_path(self) -> pathlib.Path:
        return pathlib.Path(str(self.lineEdit_5.text()))

    @property
    def channel_a(self) -> list[int]:
        s: str = str(self.lineEdit.text())
        if s:
            return [int(x) for x in s.replace(',', ' ').split()]
        elif isinstance(self.tttr, tttrlib.TTTR):
            return list(map(int, self.tttr.get_used_routing_channels()))
        return []

    @property
    def channel_b(self) -> list[int]:
        s: str = str(self.lineEdit_2.text())
        if s:
            return [int(x) for x in s.replace(',', ' ').split()]
        elif isinstance(self.tttr, tttrlib.TTTR):
            return list(map(int, self.tttr.get_used_routing_channels()))
        return []

    @property
    def filter_file(self) -> str:
        return str(self.lineEdit_4.text())

    @property
    def filter_enabled(self) -> bool:
        return pathlib.Path(self.filter_file).exists()

    @property
    def correlation_nbins(self) -> int:
        return int(self.spinBox_2.value())

    @property
    def correlation_ncasc(self) -> int:
        return int(self.spinBox_3.value())

    @property
    def correlation_is_fine(self) -> bool:
        return bool(self.checkBox_2.isChecked())

    @property
    def correlation_nsplits(self) -> int:
        return int(self.spinBox.value())

    @property
    def target_path(self) -> pathlib.Path:
        return pathlib.Path(str(self.lineEdit_5.text()))

    @property
    def microtime_range_a(self) -> list[tuple[int, int]]:
        s: str = str(self.lineEdit_6.text())
        return self.get_microtime_ranges(s)

    @property
    def microtime_range_b(self) -> list[tuple[int, int]]:
        s: str = str(self.lineEdit_7.text())
        return self.get_microtime_ranges(s)

    def get_microtime_ranges(self, s) -> typing.List[typing.Tuple[int, int]] | None:
        chisurf.logging.log(0, "WizardTTTRCorrelator::get_microtime_ranges")
        if not s:
            chisurf.logging.log(0, "::microtime_ranges: Warning - Input string is empty.")
            return None

        try:
            text = str(s).strip()
            if not text:
                return None

            # Allow both ';' and ',' as range separators to be more user friendly.
            segments = []
            for item in text.replace(',', ';').split(';'):
                item = item.strip()
                if item:
                    segments.append(item)

            if not segments:
                chisurf.logging.log(0, "::microtime_ranges: No usable ranges after parsing.")
                return None

            ranges: typing.List[typing.Tuple[int, int]] = []
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue

                # Support either ":" or "-" between min and max while allowing
                # negative bounds such as "-1000:2000" or "-1000-2000".
                if ':' in seg:
                    a_txt, b_txt = seg.split(':', 1)
                else:
                    # Fallback for legacy "a-b" syntax; use the last '-' so that
                    # leading '-' signs in negative numbers are preserved.
                    pos = seg.rfind('-')
                    if pos <= 0:
                        # Single value like "-1000"  treat as [-1000, -1000]
                        a_txt = seg
                        b_txt = seg
                    else:
                        a_txt = seg[:pos]
                        b_txt = seg[pos + 1 :]

                a = int(a_txt.strip())
                b = int(b_txt.strip())
                if a <= b:
                    ranges.append((a, b))
                else:
                    ranges.append((b, a))

            return ranges if ranges else None

        except (ValueError, TypeError):
            chisurf.logging.log(1, "::microtime_ranges: Invalid values in microsecond ranges.")
            return None

    def update_plots(self):
        chisurf.logging.log(0, 'WizardTTTRCorrelator::Updating plots')
        self.pw_fcs.clear()
        if self.is_correlated:
            for i, cor in enumerate(self.correlations):
                pen = pg.mkPen(chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex'], width=1)
                self.plot_item_fcs.plot(x=cor['x'], y=cor['y'], pen=pen)

    def read_tttrs(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::read_tttrs")
        fn = self.current_tttr_filename
        if fn:
            if pathlib.Path(fn).exists():
                n = len(self.settings['tttr_filenames'])
                self.spinBox_4.setMaximum(n - 1)
                self.comboBox.setEnabled(False)
                self.tttr = tttrlib.TTTR(fn, self.filetype)
                header = self.tttr.get_header()
                s = header.json
                d = json.loads(s)
                self.settings['header'] = d
                self.update_plots()

    def update_output_path(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::update_output_path")
        if len(self.channel_a) > 0 and len(self.channel_b) > 0:
            cha = ','.join([str(x) for x in self.channel_a])
            chb = ','.join([str(x) for x in self.channel_b])
            chs = cha + '-' + chb
        else:
            chs = 'All'
        s = pathlib.Path('cr5') / f'{chs}'
        self.lineEdit_5.setText(s.as_posix())

    def update_parameter(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::update_parameter")
        self.settings['correlation']['is_fine'] = self.correlation_is_fine
        self.settings['correlation']['ncasc'] = self.correlation_ncasc
        self.settings['correlation']['nbins'] = self.correlation_nbins
        self.settings['correlation']['nsplits'] = self.correlation_nsplits
        self.settings['correlation']['channel_a'] = self.channel_a
        self.settings['correlation']['channel_b'] = self.channel_b
        self.settings['correlation']['filter'] = self.filter_file

        # Reset correlation flag when parameters change
        self.is_correlated = False

        self.update_plots()
        self.update_output_path()

    def onClearFiles(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::onClearFiles")
        self.settings['tttr_filenames'].clear()
        self.comboBox.setEnabled(True)
        self.lineEdit.clear()
        self.tttr = None

        # Reset correlation flag
        self.is_correlated = False

    def split_array(self, tttr, n):
        chisurf.logging.log(0, "WizardTTTRCorrelator::split_array")
        chunk_size = len(tttr) // n
        chunks = [tttr[i * chunk_size: (i + 1) * chunk_size] for i in range(n)]
        return chunks

    def get_correlation_settings(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::Getting correlation settings")
        d = {
            "n_bins": self.correlation_nbins,
            "n_casc": self.correlation_ncasc,
            "make_fine": self.correlation_is_fine
        }
        chisurf.logging.log(0, "Correlation settings:", d)
        return d

    def save_correlations(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::saving correlations to files")
        # If disabled, skip writing per-chunk files (direct TTTR mode)
        if not getattr(self, 'save_chunks_to_disk', True):
            return
        # Ensure analysis folder is set even for direct TTTR correlation
        try:
            self.ensure_analysis_folder_default()
        except Exception:
            pass
        output_folder = self.analysis_folder / self.output_path
        output_folder.mkdir(parents=True, exist_ok=True)
        # Save each chunk as a .cor text file with columns: tau, G, suren (duration, count_rate), ey (zeros)
        for i, cor in enumerate(self.correlations):
            try:
                x = np.array(cor.get('x', []))
                y = np.array(cor.get('y', []))
                duration = float(cor.get('duration', 0.0))
                # Derive mean count rate from channel counts
                try:
                    ca = float(cor.get('channel_a', {}).get('counts', 0.0))
                    cb = float(cor.get('channel_b', {}).get('counts', 0.0))
                    # Mean count rate in kHz (kristine format expects kHz)
                    # Average count rate per channel: (total photons / 2) / duration / 1000
                    count_rate = (ca + cb) / 2.0 / duration / 1000.0 if duration > 0 else 0.0
                    print(f"Chunk {i}: duration={duration}s, counts={ca+cb}, count_rate={count_rate}kHz")
                except Exception:
                    count_rate = 0.0
                suren = np.zeros_like(x)
                if suren.size > 0:
                    suren[0] = duration
                if suren.size > 1:
                    suren[1] = count_rate
                ey = np.zeros_like(x)
                mat = np.vstack([x, y, suren, ey])
                cor_path = output_folder / f'chnk-{i:04}.cor'
                # Use native path string for Windows compatibility
                # Format with 5 significant digits, suppress scientific notation for small numbers
                np.savetxt(str(cor_path), mat.T, delimiter='\t', fmt='%.5g')
            except Exception:
                # Best effort: continue saving remaining chunks
                continue

    def correlate_data(self):
        chisurf.logging.log(0, "WizardTTTRCorrelator::Correlate data")

        # Ensure default analysis folder if missing (use parent of first TTTR file)
        try:
            self.ensure_analysis_folder_default()
        except Exception:
            pass

        n_chunks = self.correlation_nsplits
        ch1 = self.channel_a
        ch2 = self.channel_b
        chisurf.logging.log(0, "ch1", ch1)
        chisurf.logging.log(0, "ch2", ch2)
        chisurf.logging.log(0, "n_chunks", n_chunks)
        chisurf.logging.log(0, "self.tttr:", self.tttr)

        # **Handle empty tttr case**
        if self.tttr is None or len(self.tttr) == 0:
            chisurf.logging.log(1, "Warning: No TTTR data available for correlation.")

            # **Display a message box to the user**
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("No Photons Selected")
            msg_box.setText("No photons selected for correlation. Please load data before continuing.")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.exec_()

            return  # Stop execution

        correlation_settings = self.get_correlation_settings()
        self.correlations.clear()

        # Create a progress dialog
        progress = QtWidgets.QProgressDialog("Computing correlations...", "Cancel", 0, n_chunks, self)
        progress.setWindowTitle("Correlation Progress")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowFlags(progress.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        progress.show()

        for i, tttr in enumerate(self.split_array(self.tttr, n_chunks)):
            if progress.wasCanceled():
                chisurf.logging.log(1, "Correlation process was canceled by the user.")
                break

            # **Handle empty chunk case**
            if tttr is None or len(tttr.macro_times) == 0:
                chisurf.logging.log(1, f"Warning: Skipping chunk {i} due to empty TTTR data.")
                continue

            print(f"Processing chunk {i}...")
            print(f"Computing masks for chunk {i}...")
            t = tttr.macro_times

            mask_a = tttrlib.TTTRMask()
            mask_b = tttrlib.TTTRMask()

            mask_a.select_channels(tttr, ch1, mask=True)
            mask_b.select_channels(tttr, ch2, mask=True)

            # Get masks as numpy arrays (zero-copy views)
            m_a = mask_a.mask.astype(bool)
            m_b = mask_b.mask.astype(bool)

            if self.microtime_range_a:
                mask_mt_a = tttrlib.TTTRMask()
                mask_mt_a.select_microtime_ranges(tttr, self.microtime_range_a)
                mask_mt_a.flip()
                m_a = np.logical_and(m_a, mask_mt_a.mask.astype(bool))

            if self.microtime_range_b:
                mask_mt_b = tttrlib.TTTRMask()
                mask_mt_b.select_microtime_ranges(tttr, self.microtime_range_b)
                mask_mt_b.flip()
                m_b = np.logical_and(m_b, mask_mt_b.mask.astype(bool))

            w1 = np.array(m_a, dtype=np.float64)
            w2 = np.array(m_b, dtype=np.float64)

            sw1 = w1.sum()
            sw2 = w2.sum()

            # macro_time_resolution is in seconds, multiply by 1000 to get milliseconds
            dT = tttr.header.macro_time_resolution * 1000.0

            # **Handle empty macro_times to prevent IndexError**
            if len(t) == 0:
                chisurf.logging.log(1, f"Warning: Skipping chunk {i} due to missing macro_times.")
                continue
            
            # Compute duration more robustly using percentiles to avoid outliers
            # Use 0.1% and 99.9% percentiles instead of first/last photon
            if len(t) > 100:
                t_start = np.percentile(t, 0.1)
                t_end = np.percentile(t, 99.9)
            else:
                # For small datasets, use first and last photon
                t_start = t[0]
                t_end = t[-1]
            dur = (t_end - t_start) * dT  # duration in milliseconds

            print(f"Computing correlation for chunk {i}...")
            print(f"Duration: {dur}, start: {t_start}, end: {t_end}, dT: {dT}")

            if sw1 > 0.0 and sw2 > 0.0:
                correlator = tttrlib.Correlator(**correlation_settings)
                correlator.set_macrotimes(t, t)
                correlator.set_weights(w1, w2)
                x = correlator.x_axis * dT
                if self.correlation_is_fine:
                    n_microtime_channels = tttr.get_number_of_micro_time_channels()
                    mt = tttr.micro_times
                    correlator.set_microtimes(mt, mt, n_microtime_channels)
                    x /= (tttr.header.micro_time_resolution / 1000.0)
                print(f"Chunk {i}: sw1={sw1}, sw2={sw2}, dur_ms={dur}, dur_s={dur/1000.0}, dT={dT}")
                d = {
                    'x': x.tolist(),
                    'y': correlator.correlation.tolist(),
                    'correlation_settings': correlation_settings,
                    'analysis_folder': self.analysis_folder.as_posix(),
                    'chunk': i,
                    'duration': dur / 1000.0, # duration in seconds
                    'channel_a': {
                        'channels': ch1,
                        'microtime_range': self.microtime_range_a,
                        'counts': sw1
                    },
                    'channel_b': {
                        'channels': ch2,
                        'microtime_range': self.microtime_range_b,
                        'counts': sw2
                    }
                }
                self.correlations.append(d)
                
                # Update plot immediately after computing each correlation
                self.is_correlated = True
                pen = pg.mkPen(chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex'], width=1)
                self.plot_item_fcs.plot(x=d['x'], y=d['y'], pen=pen)
            else:
                chisurf.logging.log(1, "Warning: No photons to correlate with.")

            # Update progress bar and keep it on top
            progress.setValue(i + 1)
            progress.raise_()
            progress.activateWindow()
            QtWidgets.QApplication.processEvents()  # Keeps UI responsive

        progress.close()
        self.is_correlated = True
        self.save_correlations()

    def ensure_analysis_folder_default(self):
        """
        Ensure lineEdit_3 (analysis folder) is populated. If empty and TTTR files
        are known, set it to the parent folder of the first TTTR file.
        """
        try:
            txt = str(self.lineEdit_3.text()).strip()
        except Exception:
            txt = ""
        if (not txt) and isinstance(self.settings, dict):
            files = self.settings.get('tttr_filenames', []) or []
            if files:
                try:
                    first_parent = pathlib.Path(files[0]).resolve().parent
                    self.lineEdit_3.setText(first_parent.as_posix())
                except Exception:
                    pass

    def load_tttr_files(self, filenames: typing.List[str], filetype: typing.Optional[str] = None):
        """
        Load a list of TTTR files directly (and optionally .bst burst-id files),
        concatenate them into a single TTTR object, and update internal state.

        Behavior:
        - Plain TTTR files are opened normally (optionally with the provided filetype).
        - .bst files are parsed to obtain start/stop index ranges; the corresponding
          TTTR file is searched in the same folder or up to three parent folders.
          The TTTR events are then restricted to the union of the provided ranges.

        File type selection:
        - If filetype is a string (selected in DetectorWizardPage), pass it to tttrlib.TTTR.
        - If filetype is None (Auto), rely on tttrlib's internal auto-detection by omitting the argument.
        """
        # Split into .bst and non-.bst paths
        bst_files = []
        plain_files = []
        for fn in filenames or []:
            try:
                if str(fn).lower().endswith('.bst'):
                    bst_files.append(str(fn))
                else:
                    plain_files.append(str(fn))
            except Exception:
                continue

        # Resolve .bst files to (tttr_path, idx_array)
        resolved_from_bst: typing.Dict[str, typing.List[typing.Tuple[int, int]]] = {}
        for bst in bst_files:
            p_bst = pathlib.Path(bst)
            if not p_bst.exists() or not p_bst.is_file():
                continue
            # Determine the referenced TTTR filename (basename contains extension)
            base_with_ext = p_bst.name[:-4]  # strip trailing '.bst'
            # Search current dir and up to three parents for the TTTR file
            candidates = [p_bst.parent]
            try:
                if p_bst.parent.parent:
                    candidates.append(p_bst.parent.parent)
                if p_bst.parent.parent.parent:
                    candidates.append(p_bst.parent.parent.parent)
                if p_bst.parent.parent.parent.parent:
                    candidates.append(p_bst.parent.parent.parent.parent)
            except Exception:
                pass
            tttr_path = None
            for folder in candidates:
                cand = folder / base_with_ext
                if cand.exists() and cand.is_file():
                    tttr_path = cand
                    break
            if tttr_path is None:
                # Could not locate TTTR file for this .bst; skip gracefully
                chisurf.logging.log(1, f"Could not resolve TTTR for BST: {p_bst}")
                continue
            # Parse start/stop ranges from bst file
            ranges: typing.List[typing.Tuple[int, int]] = []
            try:
                with open(p_bst, 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith('#') or line.startswith('//'):
                            continue
                        parts = line.replace(',', ' ').split()
                        if len(parts) < 2:
                            continue
                        try:
                            s = int(float(parts[0]))
                            e = int(float(parts[1]))
                            if e >= s:
                                ranges.append((s, e))
                        except Exception:
                            continue
            except Exception as e:
                chisurf.logging.log(1, f"Failed to parse BST file '{p_bst}': {e}")
                continue
            if not ranges:
                continue
            key = str(tttr_path.resolve())
            resolved_from_bst.setdefault(key, []).extend(ranges)

        # Merge overlapping/adjacent ranges per TTTR and convert to numpy indices
        bst_indices: typing.Dict[str, typing.Any] = {}
        for tttr_path, rr in resolved_from_bst.items():
            try:
                # sort ranges
                rr = sorted(rr)
                merged: typing.List[typing.Tuple[int, int]] = []
                for s, e in rr:
                    if not merged:
                        merged.append((s, e))
                    else:
                        ps, pe = merged[-1]
                        if s <= pe + 1:
                            merged[-1] = (ps, max(pe, e))
                        else:
                            merged.append((s, e))
                # Build a single index array (inclusive ranges)
                import numpy as _np
                parts = [
                    _np.arange(s, e + 1, dtype=_np.int64)
                    for s, e in merged if e >= s
                ]
                if parts:
                    bst_indices[tttr_path] = _np.concatenate(parts)
            except Exception:
                continue

        # Now construct TTTR by loading plain files and bst-resolved files, applying indices
        tttr_obj = None
        def _open_tttr(path_str: str):
            try:
                p = pathlib.Path(path_str)
                p_posix = p.as_posix()
                ext = p.suffix.lower()
                if ext == '.spc':
                    try:
                        ft_int = tttrlib.inferTTTRFileType(p_posix)
                        if ft_int is not None and ft_int >= 0:
                            return tttrlib.TTTR(p_posix, ft_int)
                    except Exception:
                        pass
                    try:
                        return tttrlib.TTTR(p_posix, 'SPC')
                    except Exception:
                        return tttrlib.TTTR(p_posix)
                if isinstance(filetype, str) and filetype.strip():
                    try:
                        return tttrlib.TTTR(p_posix, filetype)
                    except Exception:
                        pass
                try:
                    ft_int = tttrlib.inferTTTRFileType(p_posix)
                    if ft_int is not None and ft_int >= 0:
                        return tttrlib.TTTR(p_posix, ft_int)
                except Exception:
                    pass
                return tttrlib.TTTR(p_posix)
            except Exception:
                return None

        # Remove plain TTTR files that are also referenced by BST selections to avoid duplicates
        bst_tttr_set = set(bst_indices.keys())
        filtered_plain = []
        for fn in plain_files:
            try:
                if str(pathlib.Path(fn).resolve()) not in bst_tttr_set:
                    filtered_plain.append(fn)
            except Exception:
                filtered_plain.append(fn)

        # 1) Load plain TTTR files first
        for fn in filtered_plain:
            p = pathlib.Path(fn)
            if not p.exists() or not p.is_file():
                continue
            tt = _open_tttr(str(p))
            if tt is None:
                continue
            if tttr_obj is None:
                tttr_obj = tt
            else:
                tttr_obj.append(tt)

        # 2) Load TTTR files resolved from BST with index restriction
        for tttr_path, idx in bst_indices.items():
            tt = _open_tttr(tttr_path)
            if tt is None:
                continue
            try:
                # Clip indices to valid range to avoid selection dimension warnings
                try:
                    n_events = len(tt)
                except Exception:
                    n_events = None
                if n_events is not None:
                    import numpy as _np
                    idx = _np.asarray(idx, dtype=_np.int64)
                    if idx.size == 0:
                        continue
                    idx = idx[_np.logical_and(idx >= 0, idx < n_events)]
                    if idx.size == 0:
                        continue
                tt = tt[idx]
            except Exception:
                # If advanced indexing not supported, fall back to sequential append of slices
                try:
                    import numpy as _np
                    if idx is not None and idx.size > 0:
                        # As a last resort, build via contiguous chunks
                        splits = _np.where(_np.diff(idx) > 1)[0]
                        start = 0
                        parts = []
                        for s in splits:
                            parts.append(tt[idx[start:s+1]])
                            start = s + 1
                        parts.append(tt[idx[start:]])
                        if parts:
                            first = parts[0]
                            for part in parts[1:]:
                                first.append(part)
                            tt = first
                except Exception:
                    pass
            if tttr_obj is None:
                tttr_obj = tt
            else:
                tttr_obj.append(tt)

        # Update visible filenames to underlying TTTR files (not the .bst wrappers)
        visible_files = filtered_plain + list(bst_indices.keys())
        self.settings.setdefault('tttr_filenames', [])
        self.settings['tttr_filenames'] = visible_files
        self.tttr = tttr_obj
        # Prefer analysis folder from the first underlying TTTR path if available
        try:
            if visible_files:
                first_parent = pathlib.Path(visible_files[0]).resolve().parent
                self.lineEdit_3.setText(first_parent.as_posix())
        except Exception:
            pass

    def open_sl5(self, filename: str) -> tttrlib.TTTR | None:
        chisurf.logging.log(0, 'WizardTTTRCorrelator::open_sl5:', filename)
        data = dict()
        try:
            with io.open_maybe_zipped(filename) as fp:
                data.update(json.load(fp))
        except Exception as e:
            chisurf.logging.log(1, f"Failed to read selection file {filename}: {e}")
            return None
        tttr_filename = self.analysis_folder / pathlib.Path(data.get('filename', ''))
        tttr_filetype = data.get('filetype')
        f = chisurf.fio.decompress_numpy_array(data.get('filter'))
        idx = np.where(f > 0)[0] if f is not None else None
        if not tttr_filename.exists():
            chisurf.logging.log(1, f"TTTR source file does not exist: {tttr_filename}")
            return None
        chisurf.logging.log(0, 'tttr_filetype: ', tttr_filetype)
        tttr = tttrlib.TTTR(tttr_filename.as_posix(), tttr_filetype)
        if idx is not None:
            tttr = tttr[idx]
        return tttr

    def open_selections(self, filenames: typing.List[pathlib.Path]) -> tttrlib.TTTR | None:
        chisurf.logging.log(0, "WizardTTTRCorrelator::open_selections:", filenames)
        if not filenames:
            chisurf.logging.log(1, "No selection files provided to open_selections.")
            return None
        first = self.open_sl5(str(filenames[0]))
        if first is None:
            chisurf.logging.log(1, f"Failed to open first selection file: {filenames[0]}")
            return None
        self.tttr = first
        for filename in filenames[1:]:
            tttr_part = self.open_sl5(str(filename))
            if tttr_part is not None:
                self.tttr.append(tttr_part)
        chisurf.logging.log(0, "tttr", self.tttr)
        return self.tttr

    def open_analysis_folder(self, folder: pathlib.Path = None):
        chisurf.logging.log(0, "WizardTTTRCorrelator::open_analysis_folder")
        if folder is None:
            folder = self.analysis_folder / 'sl5'
        if not folder.exists():
            chisurf.logging.log(1, f"Analysis folder does not exist: {folder}")
            return
        selected_files = sorted(list(folder.glob('*.json.gz')))
        chisurf.logging.log(0, 'Opening analysis folder')
        chisurf.logging.log(0, list(selected_files))
        if not selected_files:
            chisurf.logging.log(1, f"No selection files (*.json.gz) found in: {folder}")
            return
        self.open_selections(selected_files)

    @chisurf.gui.decorators.init_with_ui("tttr_correlator.ui")
    def __init__(
            self,
            ncasc: int = None,
            nbins: int = None,
            nsplits: int = None,
            is_fine: bool = None,
            channel_a: str = "",
            channel_b: str = "",
            filter_file: str = "",
            analysis_folder: str = "",
            output_path: str = "",
            microtime_range_a: str = "",
            microtime_range_b: str = "",
            *args,
            **kwargs
    ):
        """
        Initializes the TTTR Correlation Wizard with optional parameters.

        Parameters:
        ----------
        ncasc : int, optional
            Number of cascades in correlation. If None, the value is taken from 
            cs_settings['correlator']['number_of_cascades'] at runtime.
        nbins : int, optional
            Number of bins for correlation. If None, the value is taken from 
            cs_settings['correlator']['B'] at runtime.
        nsplits : int, optional
            Number of data splits for correlation. If None, the value is taken from 
            cs_settings['correlator']['split'] at runtime.
        is_fine : bool, optional
            Whether to use fine correlation. If None, the value is taken from 
            cs_settings['correlator']['fine'] at runtime.
        channel_a : str, optional
            Comma-separated list of channels for detector A, default is "" (empty).
        channel_b : str, optional
            Comma-separated list of channels for detector B, default is "" (empty).
        filter_file : str, optional
            Path to the filter file, default is "" (none).
        analysis_folder : str, optional
            Path to the analysis folder, default is "".
        output_path : str, optional
            Path to save correlation results, default is "".
        microtime_range_a : str, optional
            Semi-colon separated microtime ranges for channel A (e.g., "0-100;200-300").
        microtime_range_b : str, optional
            Semi-colon separated microtime ranges for channel B (e.g., "50-150;250-350").

        Notes:
        ------
        - UI elements are set based on the provided arguments.
        - The correlation flag (`self.is_correlated`) is invalidated when any parameter is modified.
        - Default values for correlation parameters are taken from chisurf.settings.cs_settings at runtime,
          allowing them to reflect any changes to settings that occur during runtime.
        """

        self.setTitle("Correlator")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)

        self.tttr: tttrlib.TTTR = None
        self.settings: dict = dict()
        self.settings['correlation'] = dict()
        self.settings['tttr_filenames'] = []
        self.correlations = list()
        # Control whether per-chunk JSON files are written to disk
        self.save_chunks_to_disk = True

        # Flag to track correlation status
        self.is_correlated = False

        # FCS presets (from fcs_channel_setups.json)
        self._fcs_presets = []
        self._fcs_preset_detectors = {}
        self._fcs_preset_corr = {}

        _setup_ui(self)

        # Apply UI modifications from arguments
        self._apply_initial_parameters(
            ncasc, nbins, nsplits, is_fine, channel_a, channel_b,
            filter_file, analysis_folder, output_path, microtime_range_a, microtime_range_b
        )

        # Force update of UI elements with values from settings
        # This ensures that any default values from the UI file are overridden
        self.spinBox_2.setValue(int(chisurf.settings.cs_settings['correlator']['B']))
        self.spinBox_3.setValue(int(chisurf.settings.cs_settings['correlator']['number_of_cascades']))
        self.spinBox.setValue(int(chisurf.settings.cs_settings['correlator']['split']))
        self.checkBox_2.setChecked(bool(chisurf.settings.cs_settings['correlator']['fine']))

        # Ensure parameters are updated after setting them
        self.update_parameter()

    # ---- FCS presets ----------------------------------------------

    def load_fcs_presets(self, setup_name, detectors) -> None:
        """Load presets for a detector setup into the preset combobox."""
        if getattr(self, 'comboBox_fcs_preset', None) is None:
            return
        cfg = load_fcs_channel_setups()
        setups = cfg.get('setups', {}) if isinstance(cfg, dict) else {}
        block = setups.get(setup_name or "", {}) if isinstance(setups, dict) else {}
        pairs = block.get('pairs', []) if isinstance(block, dict) else []
        if not isinstance(pairs, list):
            pairs = []
        self._fcs_presets = pairs
        self._fcs_preset_detectors = detectors or {}
        self._fcs_preset_corr = block.get('correlator', {}) if isinstance(block, dict) else {}
        cb = self.comboBox_fcs_preset
        try:
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("")
            for p in self._fcs_presets:
                try:
                    cha = str(p.get('channel_a', ''))
                    chb = str(p.get('channel_b', ''))
                    nm = str(p.get('name', ''))
                except Exception:
                    continue
                if not nm:
                    if cha and chb:
                        nm = f"{cha}×{chb}" if cha != chb else f"{cha}_ACF"
                    else:
                        nm = "(unnamed)"
                cb.addItem(nm)
        finally:
            try:
                cb.blockSignals(False)
            except Exception:
                pass

    def _on_fcs_preset_changed(self, index: int) -> None:
        if index <= 0 or not self._fcs_presets:
            return
        try:
            pair = self._fcs_presets[index - 1]
        except Exception:
            return
        dets = self._fcs_preset_detectors or {}
        try:
            cha_name = str(pair.get('channel_a', ''))
            chb_name = str(pair.get('channel_b', ''))
        except Exception:
            return
        da = dets.get(cha_name, {}) if isinstance(dets, dict) else {}
        db = dets.get(chb_name, {}) if isinstance(dets, dict) else {}
        chs_a = da.get('chs', []) or []
        chs_b = db.get('chs', []) or chs_a
        if chs_a:
            self.lineEdit.setText(','.join(map(str, chs_a)))
        if chs_b:
            self.lineEdit_2.setText(','.join(map(str, chs_b)))
        mta = da.get('micro_time_ranges', []) or []
        mtb = db.get('micro_time_ranges', []) or []
        if mta:
            self.lineEdit_6.setText(';'.join(f"{a}-{b}" for a, b in mta))
        if mtb:
            self.lineEdit_7.setText(';'.join(f"{a}-{b}" for a, b in mtb))
        corr = dict(self._fcs_preset_corr)
        pc = pair.get('correlator')
        if isinstance(pc, dict):
            corr.update(pc)
        try:
            if 'n_bins' in corr:
                self.spinBox_2.setValue(int(corr['n_bins']))
            if 'n_casc' in corr:
                self.spinBox_3.setValue(int(corr['n_casc']))
            if 'make_fine' in corr:
                self.checkBox_2.setChecked(bool(corr['make_fine']))
        except Exception:
            pass
        try:
            self.update_parameter()
            self.update_output_path()
        except Exception:
            pass

    def _apply_initial_parameters(
            self, ncasc, nbins, nsplits, is_fine, channel_a, channel_b,
            filter_file, analysis_folder, output_path, microtime_range_a, microtime_range_b
    ):
        """
        Sets initial values of UI elements based on provided parameters.
        This method ensures that the correlation flag (`self.is_correlated`) is invalidated.

        If any of the correlation parameters (ncasc, nbins, nsplits, is_fine) are None,
        their values are taken from chisurf.settings.cs_settings at runtime.
        """

        chisurf.logging.log(0, "Setting initial parameters for UI elements")

        # Always get the latest values from settings
        settings_ncasc = chisurf.settings.cs_settings['correlator']['number_of_cascades']
        settings_nbins = chisurf.settings.cs_settings['correlator']['B']
        settings_nsplits = chisurf.settings.cs_settings['correlator']['split']
        settings_is_fine = bool(chisurf.settings.cs_settings['correlator']['fine'])

        # Use provided parameters if not None, otherwise use settings
        if ncasc is None:
            ncasc = settings_ncasc
        if nbins is None:
            nbins = settings_nbins
        if nsplits is None:
            nsplits = settings_nsplits
        if is_fine is None:
            is_fine = settings_is_fine

        # Map each parameter to its corresponding UI widget
        ui_elements = {
            'ncasc': (self.spinBox_3, ncasc),
            'nbins': (self.spinBox_2, nbins),
            'nsplits': (self.spinBox, nsplits),
            'is_fine': (self.checkBox_2, is_fine),
            'channel_a': (self.lineEdit, channel_a),
            'channel_b': (self.lineEdit_2, channel_b),
            'filter_file': (self.lineEdit_4, filter_file),
            'analysis_folder': (self.lineEdit_3, analysis_folder),
            'output_path': (self.lineEdit_5, output_path),
            'microtime_range_a': (self.lineEdit_6, microtime_range_a),
            'microtime_range_b': (self.lineEdit_7, microtime_range_b),
        }

        # Apply values to UI elements
        for key, (widget, value) in ui_elements.items():
            if isinstance(widget, QtWidgets.QSpinBox):  # Numerical inputs
                widget.setValue(int(value))
            elif isinstance(widget, QtWidgets.QCheckBox):  # Checkboxes
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QLineEdit):  # Text inputs
                widget.setText(str(value))

        # Reset correlation flag since parameters were modified
        self.is_correlated = False

    def apply_detector_setup_from_page(self, detector_page):
        """
        Populate correlator combos from a DetectorWizardPage instance.
        Uses detector_page.channels() which returns a dict mapping names to
        lists of dicts with keys: 'window_range', 'detector_chs', 'micro_time_range'.
        """
        try:
            channel_defs = detector_page.channels()
        except Exception:
            channel_defs = {}
        self.populate_channel_combos(channel_defs)

        # Load FCS presets for this detector setup, if available
        try:
            settings = detector_page.get_settings()
            dets = settings.get('detectors', {}) or {}
        except Exception:
            dets = {}
        setup_name = getattr(detector_page, 'current_setup_name', None)
        self.load_fcs_presets(setup_name, dets)

    def populate_channel_combos(self, channel_defs: dict):
        """
        Fill comboBox (A) and comboBox_2 (B) with detector-window keys.
        """
        self._channel_defs = channel_defs or {}
        keys = list(self._channel_defs.keys())
        # Sort keys for consistent UI order
        try:
            keys.sort()
        except Exception:
            pass
        # Populate combos safely
        try:
            if hasattr(self, 'comboBox'):
                self.comboBox.blockSignals(True)
                self.comboBox.clear()
                self.comboBox.addItems(keys)
                if self.comboBox.count() > 0:
                    self.comboBox.setCurrentIndex(0)
                self.comboBox.blockSignals(False)
        except Exception:
            pass
        try:
            if hasattr(self, 'comboBox_2'):
                self.comboBox_2.blockSignals(True)
                self.comboBox_2.clear()
                self.comboBox_2.addItems(keys)
                if self.comboBox_2.count() > 0:
                    # If there is at least a second item, select it for B by default, else first
                    self.comboBox_2.setCurrentIndex(1 if self.comboBox_2.count() > 1 else 0)
                self.comboBox_2.blockSignals(False)
        except Exception:
            pass
        # Apply current selections to fields
        self._on_combo_changed('A')
        self._on_combo_changed('B')

    def _on_channel_text_changed(self, *_):
        """
        When the user edits the channel fields (A or B) manually, refresh parameters
        and update the suggested output filename.
        """
        try:
            self.update_parameter()
        except Exception:
            pass
        try:
            self.update_output_path()
        except Exception:
            pass

    def _on_combo_changed(self, side: str):
        """
        Update channel and microtime fields for side 'A' or 'B' when combo changes.
        """
        # Choose correct combo and targets
        if side == 'A':
            combo = getattr(self, 'comboBox', None)
            ch_edit = getattr(self, 'lineEdit', None)
            mtr_edit = getattr(self, 'lineEdit_6', None)
        else:
            combo = getattr(self, 'comboBox_2', None)
            ch_edit = getattr(self, 'lineEdit_2', None)
            mtr_edit = getattr(self, 'lineEdit_7', None)
        if combo is None or ch_edit is None or mtr_edit is None:
            return
        key = combo.currentText()
        entries = self._channel_defs.get(key)
        if not entries:
            return
        # Derive unique detector channels
        try:
            all_chs = []
            for e in entries:
                chs = e.get('detector_chs', [])
                if isinstance(chs, (list, tuple)):
                    all_chs.extend(list(chs))
            # Deduplicate preserving order
            seen = set()
            uniq = []
            for c in all_chs:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            ch_edit.setText(','.join(str(c) for c in uniq))
        except Exception:
            pass
        # Derive microtime range string by aggregating detector micro_time_range segments
        mtr_str = ""
        try:
            segs = []
            for e in entries:
                r = e.get('micro_time_range')
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    segs.append(f"{int(r[0])}-{int(r[1])}")
            mtr_str = ';'.join(segs)
        except Exception:
            mtr_str = ''
        try:
            mtr_edit.setText(mtr_str)
        except Exception:
            pass
        # Apply parameter updates
        try:
            self.update_parameter()
            self.update_output_path()
        except Exception:
            pass
