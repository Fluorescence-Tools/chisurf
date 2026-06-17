from __future__ import annotations

from pathlib import Path

import numpy as np

from .qt_stack import ensure_qt_stack


class _MaxentPriorsMixin:
    def _on_load_prior_any_clicked(self) -> None:
        """Load prior depending on current mode.

        - Lifetime mode: load lifetime prior over tau.
        - FRET mode: load distance prior over R.
        """
        if bool(self._mode_fret):
            self._on_load_dist_prior_clicked()
        else:
            self._on_load_prior_clicked()

    def _on_load_prior_clicked(self) -> None:
        _, QtWidgets, _, chisurf, _ = ensure_qt_stack()
        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load MEM prior (text/CSV)",
            start_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not fn:
            return
        try:
            arr = np.loadtxt(fn, dtype=float, ndmin=1)
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError("empty prior")
            self._prior_vec = arr
            self.label_prior_info.setText(
                f"Prior: loaded {arr.size} values from '{Path(fn).name}'"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading prior",
                f"Failed to load prior from '{fn}':\n{exc}",
            )

    def _on_load_donor_clicked(self) -> None:
        _, QtWidgets, _, chisurf, _ = ensure_qt_stack()
        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load donor spectrum (amp/tau pairs)",
            start_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not fn:
            return
        try:
            arr = np.loadtxt(fn, dtype=float, ndmin=2)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            if arr.shape[1] < 2:
                raise ValueError("Donor spectrum must contain amplitude/lifetime pairs")
            flat = arr[:, :2].astype(float).ravel()
            if flat.size % 2 != 0:
                raise ValueError("Donor spectrum must contain amplitude/lifetime pairs")
            self._donly_vec = flat
            n_pairs = flat.size // 2
            self.label_donor_info.setText(
                f"Donor spectrum: {n_pairs} amp/tau pairs from '{Path(fn).name}'"
            )
            self._update_donor_requirement_ui()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading donor spectrum",
                f"Failed to load donor spectrum from '{fn}':\n{exc}",
            )

    def _on_load_donor_from_fit_clicked(self) -> None:
        _, QtWidgets, _, chisurf, _ = ensure_qt_stack()

        try:
            from chisurf.core.models.tcspc.lifetime import LifetimeModel  # type: ignore
        except Exception:
            LifetimeModel = None  # type: ignore[assignment]
        try:
            from chisurf.core.models.tcspc.fret import FRETModel  # type: ignore
        except Exception:
            FRETModel = None  # type: ignore[assignment]

        candidates = []
        labels = []

        try:
            fit_list = list(getattr(chisurf, "fits", []) or [])
        except Exception:
            fit_list = []

        for fg in fit_list:
            try:
                for fit in fg:
                    model = getattr(fit, "model", None)
                    if model is None:
                        continue
                    arr = None
                    try:
                        if LifetimeModel is not None and isinstance(model, LifetimeModel):
                            arr = np.asarray(model.lifetime_spectrum, dtype=float).ravel()
                        elif FRETModel is not None and isinstance(model, FRETModel):
                            arr = np.asarray(model.donor_lifetime_spectrum, dtype=float).ravel()
                    except Exception:
                        arr = None
                    if arr is None or arr.size < 2 or arr.size % 2 != 0:
                        continue
                    labels.append(getattr(fit, "name", None) or "Fit")
                    candidates.append(arr)
            except Exception:
                continue

        if not candidates:
            QtWidgets.QMessageBox.information(
                self,
                "Load donor spectrum",
                "No lifetime/FRET fits with a donor spectrum were found.",
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select donor lifetime fit")
        layout = QtWidgets.QVBoxLayout(dialog)

        combo = QtWidgets.QComboBox(dialog)
        for label, arr in zip(labels, candidates):
            combo.addItem(label, arr)
        layout.addWidget(combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx = combo.currentIndex()
        if idx < 0:
            return

        try:
            arr = np.asarray(combo.itemData(idx), dtype=float).ravel()
        except Exception:
            arr = np.zeros(0, dtype=float)
        if arr.size < 2 or arr.size % 2 != 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Load donor spectrum",
                "Selected fit does not provide a valid (amplitude, lifetime) donor spectrum.",
            )
            return

        self._donly_vec = arr
        n_pairs = arr.size // 2
        label = combo.currentText()
        self.label_donor_info.setText(
            f"Donor spectrum: {n_pairs} amp/tau pairs from fit '{label}'"
        )
        self._update_donor_requirement_ui()

    def _on_load_dist_prior_clicked(self) -> None:
        _, QtWidgets, _, chisurf, _ = ensure_qt_stack()
        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load distance prior (text/CSV)",
            start_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not fn:
            return
        try:
            arr = np.loadtxt(fn, dtype=float, ndmin=1)
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError("empty prior")
            self._dist_prior_vec = arr
            self.label_dist_prior_info.setText(
                f"Distance prior: loaded {arr.size} values from '{Path(fn).name}'"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading distance prior",
                f"Failed to load distance prior from '{fn}':\n{exc}",
            )
