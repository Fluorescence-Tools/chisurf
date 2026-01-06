from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np

from chisurf.plugins.fluorescence_decay.maxent_decay.fmem.sampling import sample_mem_distribution_emcee
from .qt_stack import ensure_qt_stack


class _MaxentActionsMixin:
    def _on_help_clicked(self) -> None:
        _, QtWidgets, _, _, _ = ensure_qt_stack()

        try:
            readme_path = Path(__file__).resolve().parents[1] / "README.md"
        except Exception:
            readme_path = None

        if readme_path is None or not readme_path.is_file():
            try:
                QtWidgets.QMessageBox.information(
                    self,
                    "MaxEnt help",
                    "README.md not found next to the plugin.",
                )
            except Exception:
                pass
            return

        try:
            text = readme_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            try:
                QtWidgets.QMessageBox.critical(
                    self,
                    "MaxEnt help",
                    f"Failed to read README.md:\n{exc}",
                )
            except Exception:
                pass
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("MaxEnt MEM README")
        layout = QtWidgets.QVBoxLayout(dialog)

        text_edit = QtWidgets.QPlainTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        layout.addWidget(text_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        try:
            dialog.resize(800, 600)
        except Exception:
            pass
        dialog.exec_()

    def _on_run_clicked(self) -> None:
        try:
            self._run_mem()
        except Exception as exc:
            _, QtWidgets, _, _, _ = ensure_qt_stack()

            if str(exc) == "MEM computation cancelled":
                return
            QtWidgets.QMessageBox.critical(self, "MEM error", str(exc))

    def _on_sample_clicked(self) -> None:
        _, QtWidgets, QtCore, chisurf, _ = ensure_qt_stack()

        if self._last_result is None:
            try:
                QtWidgets.QMessageBox.warning(
                    self,
                    "MEM sampling",
                    "No MEM result available. Run MEM before sampling.",
                )
            except Exception:
                pass
            return

        result = self._last_result

        try:
            decay, dt, t = self._get_decay_and_dt()
        except Exception:
            decay = None
            dt = None
            t = self._t_axis

        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""

        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder for MEM sampling",
            start_dir,
        )
        if not target_dir:
            return

        out_dir = Path(target_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        h5_path = out_dir / "sampling.h5"
        json_path = out_dir / "sampling_project.json"

        steps_total = 500
        thin_val = 5
        walkers_val = None
        substeps_val = 50
        nprocs_val = None
        vectorized_val = None
        try:
            steps_total = int(self.spin_sample_steps.value())
        except Exception:
            pass
        try:
            thin_val = int(self.spin_sample_thin.value())
        except Exception:
            pass
        try:
            w_raw = int(self.spin_sample_walkers.value())
            walkers_val = None if w_raw <= 0 else w_raw
        except Exception:
            walkers_val = None
        try:
            substeps_val = int(self.spin_sample_substeps.value())
        except Exception:
            pass
        try:
            np_raw = int(self.spin_sample_nprocs.value())
            nprocs_val = None if np_raw <= 0 else np_raw
        except Exception:
            nprocs_val = None
        try:
            vectorized_val = bool(self.chk_sample_vectorized.isChecked())
        except Exception:
            vectorized_val = None

        class _SamplingWorker(QtCore.QObject):  # type: ignore[misc]
            finished = QtCore.Signal(dict, str)  # stats, error_message ("" on success)
            progress = QtCore.Signal(int, int)  # done, total

            @QtCore.Slot()
            def run(self):  # type: ignore[no-untyped-def]
                try:
                    try:
                        import multiprocessing as mp  # type: ignore

                        cpu_total = mp.cpu_count() or 1
                        if nprocs_val is None:
                            nprocs_eff = max(1, cpu_total - 1)
                        else:
                            nprocs_eff = max(1, min(nprocs_val, cpu_total))
                    except Exception:
                        nprocs_eff = None

                    def _progress_cb(done: int, total: int) -> bool:
                        self.progress.emit(int(done), int(total))
                        return False

                    stats = sample_mem_distribution_emcee(
                        result,
                        nwalkers=walkers_val,
                        filename=str(h5_path),
                        steps_total=steps_total,
                        thin=int(thin_val),
                        substeps=int(substeps_val),
                        progress_cb=_progress_cb,
                        nprocs=nprocs_eff,
                        csv_prefix=str(out_dir / "chisurf_sampling"),
                        vectorized=vectorized_val,
                    )
                    self.finished.emit(stats, "")
                except Exception as exc:  # pragma: no cover
                    self.finished.emit({}, str(exc))

        progress = QtWidgets.QProgressDialog(
            "Sampling MEM distribution (emcee)...",
            "Cancel",
            0,
            int(steps_total),
            self,
        )
        progress.setWindowModality(QtCore.Qt.NonModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = _SamplingWorker()
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)

        def _on_worker_progress(done: int, total: int) -> None:
            try:
                progress.setMaximum(int(total))
                progress.setValue(int(done))
            except Exception:
                pass

        def _on_worker_finished(stats: dict, error_message: str) -> None:
            try:
                progress.close()
            except Exception:
                pass
            thread.quit()
            thread.wait()

            if error_message:
                try:
                    QtWidgets.QMessageBox.critical(self, "MEM sampling error", error_message)
                except Exception:
                    pass
                return

            self._sample_stats = stats

            if decay is not None and t is not None:
                try:
                    self._update_plots_from_result(decay, t, result)
                except Exception:
                    pass

            try:
                meta = self._build_mem_meta(result, dt)
            except Exception:
                meta = {}

            project = {
                "schema": "chisurf.maxent_mem_sampling",
                "schema_version": 1,
                "created": datetime.datetime.now().isoformat(),
                "mode": "FRET" if "R" in result else "lifetime",
                "mem": meta,
                "sampling": {
                    "hdf5_file": h5_path.name,
                    "nwalkers": int(stats.get("nwalkers", 0)),
                    "steps_total": int(stats.get("steps_total", steps_total)),
                    "thin": int(stats.get("thin", 1)),
                    "substeps": int(stats.get("substeps", 1)),
                    "ndim": int(stats.get("ndim", 0)),
                    "vectorized": bool(stats.get("vectorized", False)),
                    "n_samples": int(stats.get("n_samples", 0)),
                },
            }

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(project, f, indent=2, sort_keys=True)
            except Exception:
                pass

            try:
                QtWidgets.QMessageBox.information(
                    self,
                    "MEM sampling",
                    f"Sampling completed. Saved {stats.get('n_samples', 0)} samples to folder:\n{out_dir}",
                )
            except Exception:
                pass

        worker.progress.connect(_on_worker_progress)
        worker.finished.connect(_on_worker_finished)
        thread.started.connect(worker.run)
        thread.start()

        self._sampling_thread = thread
        self._sampling_worker = worker

    def _on_save_clicked(self) -> None:
        _, QtWidgets, _, chisurf, _ = ensure_qt_stack()

        if self._last_result is None or self._t_axis is None:
            QtWidgets.QMessageBox.warning(self, "Save MEM result", "No MEM result available to save.")
            return

        result = self._last_result

        try:
            decay, dt, t = self._get_decay_and_dt()
        except Exception:
            decay = None
            dt = None
            t = self._t_axis

        try:
            fit = self._current_fit()
        except Exception:
            fit = None

        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""

        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder to save MEM result",
            start_dir,
        )
        if not target_dir:
            return

        out_dir = Path(target_dir)

        try:
            dist_axis = np.asarray(result.get("R", result.get("tau")), dtype=float).ravel()
            p = np.asarray(result["p"], dtype=float).ravel()
        except Exception:
            dist_axis = np.zeros(0, dtype=float)
            p = np.zeros(0, dtype=float)

        try:
            Fi = np.asarray(result["Fi"], dtype=float)
            y_seg = np.asarray(result["y"], dtype=float).ravel()
            sigma = np.asarray(result["sigma"], dtype=float).ravel()
            fitstart, fitstop = result["fitrange"]
            fit_seg = (Fi @ p) * sigma
            try:
                fit_add = np.asarray(result.get("fit_additive", []), dtype=float).ravel()
            except Exception:
                fit_add = np.zeros(0, dtype=float)
            if fit_add.size == fit_seg.size:
                fit_seg = fit_seg + fit_add
            wres = (y_seg - fit_seg) / sigma
        except Exception:
            Fi = None
            y_seg = None
            sigma = None
            fitstart = fitstop = None
            fit_seg = None
            wres = None

        decay_arr = None
        if decay is not None and t is not None:
            decay_arr = np.asarray(decay, dtype=float).ravel()

        try:
            lamp = self._build_irf_array(
                decay_arr.size if decay_arr is not None else result.get("y", np.zeros(0)).size,
                t,
                float(dt) if dt is not None else 1.0,
            )
        except Exception:
            lamp = None

        meta = self._build_mem_meta(result, dt)

        try:
            if dist_axis.size and p.size:
                np.savetxt(out_dir / "distribution.txt", np.column_stack([dist_axis, p]), header="axis  p")
        except Exception:
            pass

        try:
            if decay_arr is not None and t is not None:
                if fitstart is not None and fitstop is not None and fit_seg is not None:
                    full_fit = np.zeros_like(decay_arr)
                    full_fit[fitstart : fitstop + 1] = fit_seg
                    arr = np.column_stack([t, decay_arr, full_fit])
                    header = "time  decay  mem_fit"
                else:
                    arr = np.column_stack([t, decay_arr])
                    header = "time  decay"
                np.savetxt(out_dir / "decay_fit.txt", arr, header=header)
        except Exception:
            pass

        try:
            if lamp is not None and t is not None:
                lamp_arr = np.asarray(lamp, dtype=float).ravel()
                if lamp_arr.size == np.asarray(t, dtype=float).ravel().size:
                    np.savetxt(out_dir / "irf.txt", np.column_stack([t, lamp_arr]), header="time  irf")
        except Exception:
            pass

        try:
            if wres is not None and t is not None and fitstart is not None and fitstop is not None:
                t_seg = np.asarray(t, dtype=float).ravel()[fitstart : fitstop + 1]
                wres_arr = np.asarray(wres, dtype=float).ravel()
                if t_seg.size == wres_arr.size:
                    np.savetxt(out_dir / "wres.txt", np.column_stack([t_seg, wres_arr]), header="time  wres")
        except Exception:
            pass

        try:
            with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except Exception:
            pass

        QtWidgets.QMessageBox.information(self, "Save MEM result", f"Saved MEM result to:\n{out_dir}")

    def _build_mem_meta(self, result, dt):
        try:
            r0_val = float(result.get("R0", self.spin_R0.value()))
            r_min_frac = float(self.spin_R_min.value())
            r_max_frac = float(self.spin_R_max.value())
            return {
                "mode": "FRET" if "R" in result else "lifetime",
                "nu": float(result.get("nu", result.get("nu_input", 0.0))),
                "chisq": float(result.get("chisq", 0.0)),
                "S": float(result.get("S", 0.0)),
                "Q": float(result.get("Q", 0.0)),
                "fitrange": [int(x) for x in result.get("fitrange", (0, 0))],
                "dt": float(result.get("dt", dt if dt is not None else 0.0)),
                "tau0": float(result.get("tau0", self.spin_tau0.value())),
                "R0": r0_val,
                "R_min_frac": r_min_frac,
                "R_max_frac": r_max_frac,
                "R_min": r_min_frac * r0_val,
                "R_max": r_max_frac * r0_val,
                "R_points": int(self.spin_R_points.value()),
                "tau_min": float(self.spin_tau_min.value()),
                "tau_max": float(self.spin_tau_max.value()),
                "tau_bins": int(self.spin_tau_bins.value()),
                "fit_start_fraction": float(self.spin_start_frac.value()),
                "fit_nuisance": bool(self.chk_fit_nuisance.isChecked()),
            }
        except Exception:
            return {}
