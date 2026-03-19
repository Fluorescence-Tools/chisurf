# -*- coding: utf-8 -*-
"""
Archived/legacy implementations of burst processing for MLELifetimeAnalysisWizard.
Kept for documentation purposes. Not used by the application runtime.

Functions moved from wizard.py:
- process_bursts_new
- process_bursts_new2
- process_bursts_old
- process_bursts_new3

Note: These functions reference methods/attributes of MLELifetimeAnalysisWizard.
They are provided here verbatim imports via relative import from wizard will not
bind automatically. This module is for reading/reference only.
"""
from __future__ import annotations

# This file intentionally left as a placeholder to keep history.
# The full historical implementations can be retrieved from VCS history if needed.

class OldStuff:
    def process_bursts_new2(self):
        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        self.stop_processing = False
        total_bursts = len(self.df_bursts)
        progress = QProgressDialog("Processing bursts...", "Cancel", 0, total_bursts, self)
        progress.setWindowTitle("Processing bursts")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.show()

        irf_cache, bg_cache = self._build_irf_bg_cache()
        settings_cache = {det: self._ensure_channel_state(det) for det in self.channel_definer.detectors.keys()}

        # single global binning (you said it’s uniform)
        det_mbs = {int(st['micro_time_binning']) for st in settings_cache.values()}
        global_mb = int(next(iter(det_mbs))) if len(det_mbs) == 1 else int(self.micro_time_binning)

        # windows and channel LUTs per detector
        window_cache = {}
        channels_cache = {}
        rc_max_seen = 0
        for det, info in self.channel_definer.detectors.items():
            st = settings_cache[det]
            sb = int(st['micro_time_start']);
            eb = int(st['micro_time_stop'])
            if eb <= sb:
                sb, eb = map(int, self.micro_time_range)
            window_cache[det] = (sb, eb)

            chs = info.get('chs', [])
            pchs = chs[::2] if len(chs) >= 2 else chs
            schs = chs[1::2] if len(chs) >= 2 else chs
            pchs = np.asarray(pchs, dtype=int)
            schs = np.asarray(schs, dtype=int)
            channels_cache[det] = (pchs, schs)
            if len(chs):
                rc_max_seen = max(rc_max_seen, int(np.max(chs)))

        # one fitter per detector (IRF/BG fixed)
        fitters = {
            det: tttrlib.Fit23(
                dt=st['dt'],
                irf=irf_cache[det],
                background=bg_cache[det],
                period=st['excitation_period'],
                g_factor=st['g_factor'],
                l1=st['l1'],
                l2=st['l2'],
                p2s_twoIstar_flag=st['p2s_twoIstar'],
                soft_bifl_scatter_flag=st['BIFL_scatter']
            ) for det, st in settings_cache.items()
        }

        results = []
        metrics = ['2I* ', 'Tau', 'gamma', 'r0', 'rho', 'BIFL scatter?', '2I*: P+2S?', 'r Scatter', 'r Experimental']

        def default_record(fname, det, cp_sum=0, cs_sum=0):
            color = det.lower()
            rec = {
                'First File': fname,
                'Detector': det,
                'Ng-p-all': int(cp_sum),
                'Ng-s-all': int(cs_sum),
                f'Number of Photons (fit window) ({color})': int(cp_sum + cs_sum)
            }
            for m in metrics:
                rec[f'{m} ({color})'] = float('nan')
            results.append(rec)

        # progress helper
        done = 0

        def pump():
            nonlocal done
            done += 1
            if (done % 25) == 0:
                QtWidgets.QApplication.processEvents()
            progress.setValue(done)

        # group bursts by file for cache locality
        for fname, df_file in self.df_bursts.groupby('First File', sort=False):
            if self.stop_processing or progress.wasCanceled():
                break
            key = Path(fname).stem
            tttr = self.tttrs.get(key)
            if tttr is None:
                # emit defaults for all bursts in this file
                for _ in range(len(df_file)):
                    for det in self.channel_definer.detectors.keys():
                        default_record(fname, det, -1, -1);
                        pump()
                continue

            # per-file arrays
            rc_full = np.asarray(tttr.routing_channels)
            mt_full = np.asarray(tttr.micro_times)
            mt_bins_full = (mt_full // global_mb).astype(np.int32, copy=False) if global_mb > 1 else mt_full.astype(
                np.int32, copy=True)

            # build detector LUTs once per file
            rc_max = int(rc_full.max()) if rc_full.size else rc_max_seen
            det_luts = {}
            for det, (pchs, schs) in channels_cache.items():
                is_p = np.zeros(rc_max + 1, dtype=bool)
                is_s = np.zeros(rc_max + 1, dtype=bool)
                if pchs.size: is_p[pchs] = True
                if schs.size: is_s[schs] = True
                det_luts[det] = (is_p, is_s)

            # preallocate decay buffers per detector
            decay_buf = {det: np.empty(irf_cache[det].size, dtype=np.float64) for det in
                         self.channel_definer.detectors.keys()}
            half_len = {det: irf_cache[det].size // 2 for det in self.channel_definer.detectors.keys()}
            do_shift = int(self.shift) if self.shift else 0

            # iterate bursts of this file (fast path: itertuples)
            for first_ph, last_ph in df_file[['First Photon', 'Last Photon']].itertuples(index=False, name=None):
                if self.stop_processing or progress.wasCanceled():
                    break
                if first_ph < 0 or last_ph < 0:
                    for det in self.channel_definer.detectors.keys():
                        default_record(fname, det, -1, -1)
                    pump()
                    continue

                sl = slice(int(first_ph), int(last_ph))
                rc_slice = rc_full[sl]
                mt_bins = mt_bins_full[sl]

                # per detector
                for det in self.channel_definer.detectors.keys():
                    st = settings_cache[det]
                    sb, eb = window_cache[det]
                    n = half_len[det]
                    is_p_lut, is_s_lut = det_luts[det]

                    # one-pass histogram for P/S
                    cp_u32, cs_u32 = self._hist2_split(mt_bins, rc_slice, is_p_lut, is_s_lut, n)

                    # quick threshold
                    cp_sum = int(cp_u32.sum());
                    cs_sum = int(cs_u32.sum())
                    if (cp_sum + cs_sum) < int(st['min_photons']):
                        default_record(fname, det, cp_sum, cs_sum)
                        continue

                    # integer shift on cs then windowed copy into decay buffer
                    if do_shift:
                        cs_u32 = np.roll(cs_u32, do_shift)

                    d = decay_buf[det]
                    # zero everything then copy windowed ranges (fewer writes than zeroing cp/cs)
                    d.fill(0.0)
                    # vv slice
                    if sb < eb:
                        vv_len = min(eb, n) - max(sb, 0)
                        if vv_len > 0:
                            s0 = max(sb, 0);
                            s1 = s0 + vv_len
                            d[s0:s1] = cp_u32[s0:s1]
                    # vh slice
                    off = n
                    if sb < eb:
                        vh_len = min(eb, n) - max(sb, 0)
                        if vh_len > 0:
                            s0 = max(sb, 0);
                            s1 = s0 + vh_len
                            d[off + s0: off + s1] = cs_u32[s0:s1]

                    # cast once to float64 already done (d is float64)
                    fitter = fitters[det]
                    res = fitter(data=d, initial_values=st['initial_x0'], fixed=st['fixed_flags'])
                    color = det.lower()
                    results.append({
                        'First File': fname,
                        'Detector': det,
                        'Ng-p-all': cp_sum,
                        'Ng-s-all': cs_sum,
                        f'Number of Photons (fit window) ({color})': cp_sum + cs_sum,
                        f'2I*  ({color})': res.get('twoIstar', 0.0),
                        f'Tau ({color})': res['x'][0],
                        f'gamma ({color})': res['x'][1],
                        f'r0 ({color})': res['x'][2],
                        f'rho ({color})': res['x'][3],
                        f'BIFL scatter? ({color})': int(st['BIFL_scatter']),
                        f'2I*: P+2S? ({color})': int(st['p2s_twoIstar']),
                        f'r Scatter ({color})': res['x'][6] if len(res['x']) > 6 else float('nan'),
                        f'r Experimental ({color})': res['x'][7] if len(res['x']) > 7 else float('nan'),
                    })
                pump()

        try:
            progress.close()
        except Exception:
            pass

        result_df = pd.DataFrame(results)
        self._save_burst_results_fast(result_df)


    def process_bursts_old(self):
        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        # Reset stop flag before starting processing
        self.stop_processing = False

        # prepare modal progress dialog (blocks the UI)
        total_bursts = len(self.df_bursts)
        processing_progress = QProgressDialog("Processing bursts...", "Cancel", 0, total_bursts, self)
        processing_progress.setWindowTitle("Processing bursts")
        processing_progress.setWindowModality(QtCore.Qt.WindowModal)
        processing_progress.setAutoClose(False)
        processing_progress.setValue(0)
        processing_progress.show()

        # cache IRF/BG per detector (do not zero-out IRF/BG)
        irf_cache, bg_cache = self._build_irf_bg_cache()
        # cache per-detector settings once (avoid calling _ensure_channel_state inside the loops)
        settings_cache = {det: self._ensure_channel_state(det) for det in self.channel_definer.detectors.keys()}
        # precompute sanitized micro-time window and binning per detector
        window_cache = {}
        for det, st in settings_cache.items():
            sb = int(st['micro_time_start'])
            eb = int(st['micro_time_stop'])
            mb = int(st['micro_time_binning'])
            # Basic sanity: ensure 0 <= sb < eb (moved out of inner loop)
            if eb <= sb:
                sb_fix, eb_fix = self.micro_time_range
                sb, eb = int(sb_fix), int(eb_fix)
            window_cache[det] = (sb, eb, mb)

        # helper to emit a default-NaN record for fit metrics
        metrics = ['2I* ', 'Tau', 'gamma', 'r0', 'rho', 'BIFL scatter?', '2I*: P+2S?', 'r Scatter', 'r Experimental']
        results = []

        tttrs = self.tttrs
        channels = self.channel_definer.detectors.items()
        # precompute primary/secondary channel lists per detector (moved out of inner loop)
        channels_cache = {}
        for det, info in channels:
            chs = info.get('chs', []) if isinstance(info, dict) else []
            try:
                length = len(chs)
            except Exception:
                length = 0
            if length >= 2:
                pchs = chs[::2]
                schs = chs[1::2]
            else:
                # if only one or zero channels are defined, treat all as primary and copy for secondary logic
                pchs = chs
                schs = chs
            channels_cache[det] = (pchs, schs)
        # reset iterator because we consumed it above
        channels = self.channel_definer.detectors.items()

        def default_record(fname, det, cp_sum=0, cs_sum=0):
            color = det.lower()
            rec = {
                'First File': fname,
                'Detector': det,
                'Ng-p-all': cp_sum,
                'Ng-s-all': cs_sum,
                f'Number of Photons (fit window) ({color})': cp_sum + cs_sum
            }
            for m in metrics:
                rec[f'{m} ({color})'] = float('nan')
            results.append(rec)

        for idx, row in self.df_bursts.iterrows():

            # Check if processing should be stopped
            if self.stop_processing or processing_progress.wasCanceled():
                chisurf.logging.info("Burst processing stopped by user")
                break
            processing_progress.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

            fname = row['First File']
            first_ph = int(row['First Photon'])
            last_ph = int(row['Last Photon'])
            key = Path(fname).stem

            bad_index = first_ph < 0 or last_ph < 0
            tttr = tttrs.get(key)
            burst = None if bad_index else tttr[first_ph:last_ph]

            for det, info in channels:

                if bad_index:
                    default_record(fname, det, -1, -1)
                    continue

                st = settings_cache[det]
                sb, eb, mb = window_cache[det]
                irf = irf_cache[det]
                bg = bg_cache[det]
                minlength = irf.size // 2

                pchs, schs = channels_cache[det]

                # Use minlength to assert that decay matched IRF
                cp = self.filter_tttr(burst, [sb, eb], pchs) \
                    .get_microtime_histogram(mb, minlength=minlength)[0].astype(np.float64, copy=False)
                cs = self.filter_tttr(burst, [sb, eb], schs) \
                    .get_microtime_histogram(mb, minlength=minlength)[0].astype(np.float64, copy=False)

                # Early check if enough photons are available
                cp_sum = cp.sum()
                cs_sum = cs.sum()

                if (cp_sum + cs_sum < st['min_photons']):
                    chisurf.logging.debug(
                        f"Skip: {Path(fname).name} Burst: {idx} Detector: {det} NPh: {cp_sum + cs_sum} < {st['min_photons']}")
                    default_record(fname, det, cp_sum, cs_sum)
                    continue

                # apply global VH data shift before windowing
                if self.shift != 0:
                    cs = np.roll(cs, self.shift)

                # zero outside [sb:eb) in-place (safe even if sb==0 or eb==size)
                cp[:sb] = 0
                cp[eb:] = 0
                cs[:sb] = 0
                cs[eb:] = 0

                # avoid the temporary from hstack; preallocate and copy
                decay = np.empty(cp.size + cs.size, dtype=cp.dtype)
                decay[:cp.size] = cp
                decay[cp.size:] = cs

                # perform fit
                fit = tttrlib.Fit23(
                    dt=st['dt'],
                    irf=irf,
                    background=bg,
                    period=st['excitation_period'],
                    g_factor=st['g_factor'],
                    l1=st['l1'],
                    l2=st['l2'],
                    p2s_twoIstar_flag=st['p2s_twoIstar'],
                    soft_bifl_scatter_flag=st['BIFL_scatter']
                )
                res = fit(data=decay, initial_values=st['initial_x0'], fixed=st['fixed_flags'])

                # build result record
                color = det.lower()
                rec = {
                    'First File': fname,
                    'Detector': det,
                    'Ng-p-all': cp_sum,
                    'Ng-s-all': cs_sum,
                    f'Number of Photons (fit window) ({color})': cp_sum + cs_sum,
                    f'2I*  ({color})': res.get('twoIstar', 0.0),
                    f'Tau ({color})': res['x'][0],
                    f'gamma ({color})': res['x'][1],
                    f'r0 ({color})': res['x'][2],
                    f'rho ({color})': res['x'][3],
                    f'BIFL scatter? ({color})': int(st['BIFL_scatter']),
                    f'2I*: P+2S? ({color})': int(st['p2s_twoIstar']),
                    f'r Scatter ({color})': res['x'][6],
                    f'r Experimental ({color})': res['x'][7],
                }
                results.append(rec)

        result_df = pd.DataFrame(results)

        # Close the processing progress window before saving
        try:
            processing_progress.close()
        except Exception:
            pass

        # Use fast saver to write results
        self._save_burst_results_fast(result_df)


    def process_bursts_new3(self):
        import os, threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        # Reset stop flag
        self.stop_processing = False
        cancel_event = threading.Event()

        # Progress UI
        total_bursts = len(self.df_bursts)
        progress = QProgressDialog("Processing bursts...", "Cancel", 0, total_bursts, self)
        progress.setWindowTitle("Processing bursts")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.show()

        # Handle cancel: set event (workers poll it)
        def _maybe_cancel():
            if progress.wasCanceled():
                cancel_event.set()
            return cancel_event.is_set()

        # ---- Static per-detector data prepared on main thread ----
        irf_cache, bg_cache = self._build_irf_bg_cache()
        settings_cache = {det: self._ensure_channel_state(det) for det in self.channel_definer.detectors.keys()}

        # one global micro-time binning (your setup)
        det_mbs = {int(st['micro_time_binning']) for st in settings_cache.values()}
        global_mb = int(next(iter(det_mbs))) if len(det_mbs) == 1 else int(self.micro_time_binning)

        # windows (sb, eb) and channel lists; also track max routing channel id
        window_cache: dict[str, tuple[int, int]] = {}
        channels_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        rc_max_seen = 0
        for det, info in self.channel_definer.detectors.items():
            st = settings_cache[det]
            sb = int(st['micro_time_start']);
            eb = int(st['micro_time_stop'])
            if eb <= sb:
                sb, eb = map(int, self.micro_time_range)
            window_cache[det] = (sb, eb)

            chs = info.get('chs', [])
            pchs = chs[::2] if len(chs) >= 2 else chs
            schs = chs[1::2] if len(chs) >= 2 else chs
            pchs = np.asarray(pchs, dtype=int)
            schs = np.asarray(schs, dtype=int)
            channels_cache[det] = (pchs, schs)
            if len(chs):
                rc_max_seen = max(rc_max_seen, int(np.max(chs)))

        # Preload TTTRs on the main thread (avoid races in lazy loader)
        file_groups = []
        for fname, df_file in self.df_bursts.groupby('First File', sort=False):
            key = Path(fname).stem
            tttr = self.tttrs.get(key)
            file_groups.append((fname, key, df_file, tttr))

        # Helper for default records (shared)
        metrics = ['2I* ', 'Tau', 'gamma', 'r0', 'rho', 'BIFL scatter?', '2I*: P+2S?', 'r Scatter', 'r Experimental']

        def _default_record(fname, det, cp_sum=0, cs_sum=0):
            color = det.lower()
            rec = {
                'First File': fname,
                'Detector': det,
                'Ng-p-all': int(cp_sum),
                'Ng-s-all': int(cs_sum),
                f'Number of Photons (fit window) ({color})': int(cp_sum + cs_sum)
            }
            for m in metrics:
                rec[f'{m} ({color})'] = float('nan')
            return rec

        # Worker: process one file (runs in a thread)
        def _process_one_file(fname: str, key: str, df_file: pd.DataFrame, tttr_obj):
            out = []
            if cancel_event.is_set():
                return out, 0
            # If missing TTTR: emit defaults for all bursts in this file
            if tttr_obj is None:
                for _ in range(len(df_file)):
                    for det in self.channel_definer.detectors.keys():
                        out.append(_default_record(fname, det, -1, -1))
                return out, len(df_file)

            # Build per-file arrays (views)
            rc_full = np.asarray(tttr_obj.routing_channels)
            mt_full = np.asarray(tttr_obj.micro_times)

            # Global binned micro-times
            if global_mb > 1:
                mt_bins_full = (mt_full // global_mb).astype(np.int32, copy=False)
            else:
                mt_bins_full = mt_full.astype(np.int32, copy=True)

            # Per-detector LUTs (routing membership)
            rc_max = int(rc_full.max()) if rc_full.size else rc_max_seen
            det_luts = {}
            for det, (pchs, schs) in channels_cache.items():
                is_p = np.zeros(rc_max + 1, dtype=bool)
                is_s = np.zeros(rc_max + 1, dtype=bool)
                if pchs.size: is_p[pchs] = True
                if schs.size: is_s[schs] = True
                det_luts[det] = (is_p, is_s)

            # Per-detector buffers and half lengths
            half_len = {det: max(1, irf_cache[det].size // 2) for det in self.channel_definer.detectors.keys()}
            decay_buf = {det: np.empty(irf_cache[det].size, dtype=np.float64) for det in
                         self.channel_definer.detectors.keys()}
            do_shift = int(self.shift) if self.shift else 0

            # Construct Fit23 objects inside the thread (no sharing across threads)
            fitters = {
                det: tttrlib.Fit23(
                    dt=st['dt'],
                    irf=irf_cache[det],
                    background=bg_cache[det],
                    period=st['excitation_period'],
                    g_factor=st['g_factor'],
                    l1=st['l1'],
                    l2=st['l2'],
                    p2s_twoIstar_flag=st['p2s_twoIstar'],
                    soft_bifl_scatter_flag=st['BIFL_scatter']
                ) for det, st in settings_cache.items()
            }

            processed = 0
            # Fast DF iteration
            for first_ph, last_ph in df_file[['First Photon', 'Last Photon']].itertuples(index=False, name=None):
                if cancel_event.is_set():
                    break
                first_ph = int(first_ph);
                last_ph = int(last_ph)
                if first_ph < 0 or last_ph < 0:
                    for det in self.channel_definer.detectors.keys():
                        out.append(_default_record(fname, det, -1, -1))
                    processed += 1
                    continue

                sl = slice(first_ph, last_ph)
                rc_slice = rc_full[sl]
                mt_bins = mt_bins_full[sl]

                for det in self.channel_definer.detectors.keys():
                    st = settings_cache[det]
                    sb, eb = window_cache[det]
                    n = half_len[det]
                    is_p_lut, is_s_lut = det_luts[det]

                    # One-pass histogram
                    cp_u32, cs_u32 = self._hist2_split(mt_bins, rc_slice, is_p_lut, is_s_lut, n)

                    cp_sum = int(cp_u32.sum());
                    cs_sum = int(cs_u32.sum())
                    if (cp_sum + cs_sum) < int(st['min_photons']):
                        out.append(_default_record(fname, det, cp_sum, cs_sum))
                        continue

                    # Shift + window into decay buffer (float64)
                    if do_shift:
                        cs_u32 = np.roll(cs_u32, do_shift)

                    d = decay_buf[det]
                    d.fill(0.0)
                    if sb < eb:
                        s0 = max(0, sb);
                        s1 = min(n, eb)
                        if s1 > s0:
                            d[s0:s1] = cp_u32[s0:s1]
                            off = n
                            d[off + s0: off + s1] = cs_u32[s0:s1]

                    # Fit (tttrlib should release GIL here)
                    res = fitters[det](data=d, initial_values=st['initial_x0'], fixed=st['fixed_flags'])

                    color = det.lower()
                    out.append({
                        'First File': fname,
                        'Detector': det,
                        'Ng-p-all': cp_sum,
                        'Ng-s-all': cs_sum,
                        f'Number of Photons (fit window) ({color})': cp_sum + cs_sum,
                        f'2I*  ({color})': res.get('twoIstar', 0.0),
                        f'Tau ({color})': res['x'][0],
                        f'gamma ({color})': res['x'][1],
                        f'r0 ({color})': res['x'][2],
                        f'rho ({color})': res['x'][3],
                        f'BIFL scatter? ({color})': int(st['BIFL_scatter']),
                        f'2I*: P+2S? ({color})': int(st['p2s_twoIstar']),
                        f'r Scatter ({color})': res['x'][6] if len(res['x']) > 6 else float('nan'),
                        f'r Experimental ({color})': res['x'][7] if len(res['x']) > 7 else float('nan'),
                    })
                processed += 1
            return out, processed

        # Thread pool
        # Heuristic: leave one core for UI; cap to 32
        max_workers = max(1, min(32, (os.cpu_count() or 8) - 1))

        results = []
        bursts_done = 0
        try:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="burstfit") as ex:
                futures = []
                for fname, key, df_file, tttr in file_groups:
                    if _maybe_cancel(): break
                    futures.append(ex.submit(_process_one_file, fname, key, df_file, tttr))

                for fut in as_completed(futures):
                    if _maybe_cancel():
                        # Signal workers to stop & drain remaining futures
                        # (they’ll exit early on cancel_event)
                        continue
                    try:
                        out, processed = fut.result()
                        results.extend(out)
                        bursts_done += int(processed)
                        progress.setValue(min(bursts_done, total_bursts))
                        if (bursts_done % 25) == 0:
                            QtWidgets.QApplication.processEvents()
                    except Exception as e:
                        # keep going; record as NaNs for safety?
                        chisurf.logging.error(f"Worker error: {e}")
                        # You could also surface a messagebox here if preferred.
        finally:
            try:
                progress.close()
            except Exception:
                pass

        # If canceled: inform and bail out early
        if cancel_event.is_set() or self.stop_processing:
            QtWidgets.QMessageBox.information(self, "Canceled", "Burst processing was canceled.")
            return

        result_df = pd.DataFrame(results)
        self._save_burst_results_fast(result_df)


    def process_bursts_new4(self):
        import os
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        from multiprocessing import shared_memory
        from chisurf.plugins.burst.burst_mle_analysis._mp_worker import process_one_file_worker

        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        # UI
        self.stop_processing = False
        total_bursts = len(self.df_bursts)
        progress = QProgressDialog("Processing bursts...", "Cancel", 0, total_bursts, self)
        progress.setWindowTitle("Processing bursts")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.show()

        def ui_pump(k: int):
            if (k % 20) == 0:
                QtWidgets.QApplication.processEvents()

        # Per-detector constants
        irf_cache, bg_cache = self._build_irf_bg_cache()
        settings_cache = {det: self._ensure_channel_state(det) for det in self.channel_definer.detectors.keys()}
        det_order = list(self.channel_definer.detectors.keys())

        # uniform binning
        det_mbs = {int(st['micro_time_binning']) for st in settings_cache.values()}
        global_mb = int(next(iter(det_mbs))) if len(det_mbs) == 1 else int(self.micro_time_binning)

        # windows, channels, rc max
        window_cache = {}
        channels_cache = {}
        rc_max_seen = 0
        for det, info in self.channel_definer.detectors.items():
            st = settings_cache[det]
            sb = int(st['micro_time_start']);
            eb = int(st['micro_time_stop'])
            if eb <= sb: sb, eb = map(int, self.micro_time_range)
            window_cache[det] = (sb, eb)

            chs = info.get('chs', [])
            pchs = chs[::2] if len(chs) >= 2 else chs
            schs = chs[1::2] if len(chs) >= 2 else chs
            pchs = np.asarray(pchs, dtype=int);
            schs = np.asarray(schs, dtype=int)
            channels_cache[det] = (pchs, schs)
            if len(chs):
                rc_max_seen = max(rc_max_seen, int(np.max(chs)))

        # Build per-file jobs with shared memory
        jobs = []
        shm_blocks = []  # remember to unlink
        for fname, df_file in self.df_bursts.groupby('First File', sort=False):
            key = Path(fname).stem
            tttr = self.tttrs.get(key)

            if tttr is None:
                jobs.append((fname,
                             list(df_file[['First Photon', 'Last Photon']].itertuples(index=False, name=None)),
                             None, None, None, None, None, None,
                             det_order, {}, int(self.shift or 0)))
                continue

            rc_full = np.asarray(tttr.routing_channels)
            mt_full = np.asarray(tttr.micro_times)
            mt_bins_full = (mt_full // global_mb).astype(np.int32, copy=False) if global_mb > 1 else mt_full.astype(
                np.int32, copy=True)

            # compact dtype to cut bandwidth
            if rc_full.dtype != np.uint16 and int(rc_full.max(initial=0)) <= 65535:
                rc_full = rc_full.astype(np.uint16, copy=False)

            # Shared memory blocks (parent owns lifecycle)
            rc_shm = shared_memory.SharedMemory(create=True, size=rc_full.nbytes)
            np.ndarray(rc_full.shape, dtype=rc_full.dtype, buffer=rc_shm.buf)[:] = rc_full
            mt_shm = shared_memory.SharedMemory(create=True, size=mt_bins_full.nbytes)
            np.ndarray(mt_bins_full.shape, dtype=mt_bins_full.dtype, buffer=mt_shm.buf)[:] = mt_bins_full
            shm_blocks.extend([rc_shm, mt_shm])

            # Per-detector config (small objects only)
            rc_max = int(rc_full.max(initial=rc_max_seen)) if rc_full.size else rc_max_seen
            perdet_cfg = {}
            for det in det_order:
                st = settings_cache[det]
                pchs, schs = channels_cache[det]
                is_p = np.zeros(rc_max + 1, dtype=bool);
                is_s = np.zeros(rc_max + 1, dtype=bool)
                if pchs.size: is_p[pchs] = True
                if schs.size: is_s[schs] = True
                half_len = max(1, irf_cache[det].size // 2)
                perdet_cfg[det] = {
                    'sb': int(window_cache[det][0]),
                    'eb': int(window_cache[det][1]),
                    'half_len': half_len,
                    'dt': float(st['dt']),
                    'period': float(st['excitation_period']),
                    'g_factor': float(st['g_factor']),
                    'l1': float(st['l1']), 'l2': float(st['l2']),
                    'p2s_twoIstar': bool(st['p2s_twoIstar']),
                    'BIFL_scatter': bool(st['BIFL_scatter']),
                    'min_photons': int(st['min_photons']),
                    'x0': np.asarray(st['initial_x0'], dtype=np.float64),
                    'fixed': np.asarray(st['fixed_flags'], dtype=np.int32),
                    'irf': np.asarray(irf_cache[det], dtype=np.float64),
                    'bg': np.asarray(bg_cache[det], dtype=np.float64),
                    'is_p_lut': is_p, 'is_s_lut': is_s,
                }

            bursts = list(df_file[['First Photon', 'Last Photon']].itertuples(index=False, name=None))
            jobs.append((fname, bursts,
                         rc_shm.name, rc_full.shape, str(rc_full.dtype),
                         mt_shm.name, mt_bins_full.shape, str(mt_bins_full.dtype),
                         det_order, perdet_cfg, int(self.shift or 0)))

        # Processes (leave one core for UI; cap by #files)
        ctx = mp.get_context('spawn')
        max_workers = max(1, min(os.cpu_count() or 8, len(jobs)) - 1)
        results = []
        processed = 0
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futs = [ex.submit(process_one_file_worker, j) for j in jobs]
                for fut in as_completed(futs):
                    try:
                        out, nbursts = fut.result()
                    except Exception as e:
                        chisurf.logging.error(f"Worker failed: {e}")
                        out, nbursts = [], 0
                    results.extend(out)
                    processed += nbursts
                    progress.setValue(min(processed, total_bursts))
                    ui_pump(processed)
        finally:
            try:
                progress.close()
            except Exception:
                pass
            # cleanup shared memory
            for block in shm_blocks:
                try:
                    block.close();
                    block.unlink()
                except Exception:
                    pass

        if self.stop_processing or processed < total_bursts and progress.wasCanceled():
            QtWidgets.QMessageBox.information(self, "Canceled", "Burst processing was canceled.")
            return

        result_df = pd.DataFrame(results)
        self._save_burst_results_fast(result_df)
