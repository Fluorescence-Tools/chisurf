import collections.abc
import json
import typing
from pathlib import Path
from typing import Dict, Callable, Iterator

import numpy as np
import tttrlib
from qtpy import QtWidgets, QtCore, QtGui


class LazyTTTRDict(collections.abc.MutableMapping):
    """
    A dict-like that maps a key (file‐stem) → TTTR object,
    but only calls tttrlib.TTTR(path, file_type) on first access.
    """
    def __init__(
        self,
        path_map: Dict[str, Path],
        file_type_getter: Callable[[], str]
    ):
        """
        Parameters
        ----------
        path_map : Dict[str, Path]
            Maps file‐stem (no extension) → full Path to .ptu/.ht3/etc.
        file_type_getter : () → str
            A zero‐argument callable returning current TTTR file‐type (e.g. self.tttr_file_type).
        """
        self._paths = path_map
        self._cache: Dict[str, tttrlib.TTTR] = {}
        self._file_type_getter = file_type_getter
        self._warning_shown = False

    def __getitem__(self, key: str) -> tttrlib.TTTR:
        if key not in self._paths:
            raise KeyError(f"No TTTR path for key {key!r}")
        if key not in self._cache:
            path = self._paths[key]
            # Check if _file_type_getter is None
            if self._file_type_getter is None:
                if not self._warning_shown:
                    # Show a warning if a QApplication exists; otherwise, print to console.
                    if QtWidgets.QApplication.instance() is not None:
                        QtWidgets.QMessageBox.warning(
                            None,
                            "Warning",
                            "The file type getter is None. This may cause issues with TTTR file loading."
                        )
                    else:
                        print("Warning: The file type getter is None. This may cause issues with TTTR file loading.")
                    self._warning_shown = True
                # Use a default file type or try to infer it
                file_type = tttrlib.inferTTTRFileType(str(path))
            else:
                file_type = self._file_type_getter()
                # Fallback to inference if getter returned None/Auto/empty
                if not file_type or (isinstance(file_type, str) and file_type.lower() == "auto"):
                    file_type = tttrlib.inferTTTRFileType(str(path))
            # instantiate on first use
            self._cache[key] = tttrlib.TTTR(str(path), file_type)
        return self._cache[key]

    def __setitem__(self, key: str, value: tttrlib.TTTR):
        # allow manual override if you really want
        self._cache[key] = value

    def __delitem__(self, key: str):
        self._paths.pop(key, None)
        self._cache.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def add_path(self, key: str, path: Path):
        """
        Register a new TTTR file to be loaded on demand.
        """
        self._paths[key] = path

    def clear(self):
        self._paths.clear()
        self._cache.clear()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert to list (or you could serialize differently)
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return super().default(obj)


class FileListWidget(QtWidgets.QListWidget):
    """
    A QListWidget subclass that accepts file drops and maintains a list of file paths.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    file_added_callback : callable, optional
        Function to call when files are added.
    process_on_drop : bool, optional
        Whether to process files immediately on drop.
    """

    def __init__(self, parent=None, file_added_callback=None, process_on_drop=False):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.file_added_callback = file_added_callback
        self.process_on_drop = process_on_drop
        # Allow the file list to grow vertically and fill available space
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sp)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """
        Handle drag enter events to accept file URLs.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        """
        Handle drag move events to accept file URLs.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        """
        Handle drop events, extract file paths, and add them to the list.
        """
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        file_paths: typing.List[str] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if local.is_file():
                file_paths.append(str(local))
            elif local.is_dir():
                bursts = list(local.glob('**/*.bur'))
                if bursts:
                    file_paths.extend(str(f) for f in bursts)
                else:
                    for ext in tttrlib.get_supported_filetypes():
                        file_paths.extend(str(f) for f in local.glob(f'**/*{ext}'))
        file_paths.sort()
        self.blockSignals(True)
        for fp in file_paths:
            self.add_file(fp)
        self.blockSignals(False)
        if self.file_added_callback:
            self.file_added_callback()
        event.acceptProposedAction()

    def add_file(self, file_path: str):
        """
        Add a file path to the list as a checkable item.

        Parameters
        ----------
        file_path : str
            Path of the file to add.
        """
        item = QtWidgets.QListWidgetItem(file_path, self)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.addItem(item)

    def get_selected_files(self) -> typing.List[Path]:
        """
        Get the list of currently selected (checked) files.

        Returns
        -------
        List[Path]
            Paths of selected files.
        """
        return [Path(self.item(i).text()) for i in range(self.count())
                if self.item(i).checkState() == QtCore.Qt.Checked]


# --- Hyperparameter optimization utilities ---

def _relative_error(target: float, value: float, eps: float = 1e-12) -> float:
    try:
        t = float(target)
        v = float(value)
        scale = max(abs(t), eps)
        return abs(v - t) / scale
    except Exception:
        return float('inf')


def evaluate_hpo_configuration(wizard, cfg: dict, weights: dict | None = None) -> tuple[float, dict]:
    """
    Apply a hyperparameter configuration to the wizard, run a fit, and compute a loss.

    Parameters
    ----------
    wizard : MLELifetimeAnalysisWizard-like
        Object exposing the properties used in burst MLE wizard.
    cfg : dict
        Keys may include: 'irf_threshold_vv', 'irf_threshold_vh', 'shift', 'shift_sp', 'shift_ss',
        'irf_start', 'irf_stop', 'micro_time_start', 'micro_time_stop'.
    weights : dict or None
        Optional weights for components: 'tau','rho','twoIstar'. Defaults to 1 for
        model params (tau, rho) and 0.01 for twoIstar.

    Returns
    -------
    (loss, info)
        loss : float, lower is better.
        info : dict with recovered results and the twoIstar used.
    """
    # Default weights: only tau and rho are targets; gamma/r0 are not targeted
    w = {
        'tau': 1.0,
        'rho': 1.0,
        'twoIstar': 0.01
    }
    if isinstance(weights, dict):
        # Accept custom weights but ignore unknown keys in loss computation
        w.update(weights)

    try:
        # Block widget signals to avoid multiple expensive auto-fits per parameter set
        widgets = []
        try:
            widgets = [
                getattr(wizard, 'doubleSpinBox_irf_threshold_vv', None),
                getattr(wizard, 'doubleSpinBox_irf_threshold_vh', None),
                getattr(wizard, 'doubleSpinBox_shift', None),
                getattr(wizard, 'doubleSpinBox_shift_sp', None),
                getattr(wizard, 'doubleSpinBox_shift_ss', None),
                getattr(wizard, 'spinBox_irf_start', None),
                getattr(wizard, 'spinBox_irf_stop', None),
                getattr(wizard, 'spinBox_micro_time_start', None),
                getattr(wizard, 'spinBox_micro_time_stop', None),
            ]
            widgets = [w for w in widgets if w is not None]
            if hasattr(wizard, 'block_widget_signals'):
                wizard.block_widget_signals(widgets)

            # Apply micro-time range first if present
            mt_start = cfg.get('micro_time_start', None)
            mt_stop = cfg.get('micro_time_stop', None)
            if mt_start is not None and mt_stop is not None and int(mt_stop) > int(mt_start):
                wizard.micro_time_range = (int(mt_start), int(mt_stop))

            # Apply IRF parameters
            if 'irf_threshold_vv' in cfg:
                wizard.irf_threshold_vv = float(cfg['irf_threshold_vv'])
            if 'irf_threshold_vh' in cfg:
                wizard.irf_threshold_vh = float(cfg['irf_threshold_vh'])
            if 'shift' in cfg:
                wizard.shift = int(round(cfg['shift']))
            if 'shift_sp' in cfg:
                wizard.shift_sp = float(cfg['shift_sp'])
            if 'shift_ss' in cfg:
                wizard.shift_ss = float(cfg['shift_ss'])
            if 'irf_start' in cfg:
                wizard.irf_start = int(cfg['irf_start'])
            if 'irf_stop' in cfg:
                wizard.irf_stop = int(cfg['irf_stop'])
        finally:
            if hasattr(wizard, 'unblock_widget_signals') and widgets:
                wizard.unblock_widget_signals(widgets)

        # Refresh decay if window changed
        mt_start = cfg.get('micro_time_start', None)
        mt_stop = cfg.get('micro_time_stop', None)
        if mt_start is not None and mt_stop is not None and int(mt_stop) > int(mt_start):
            wizard.update_decay_of_detector()

        # Force a fresh fit build and run once
        wizard._fit = None
        wizard.update_fit()

        # Targets: current user-provided parameters
        x0, _fixed = wizard.fit_parameters
        target_tau = float(x0[0])
        target_rho = float(x0[3])

        # Recovered results
        tau_res = float(wizard.tau_result)
        gamma_res = float(wizard.gamma_result)
        r0_res = float(wizard.r0_result)
        rho_res = float(wizard.rho_result)
        twoistar = float(getattr(wizard, 'twoIstar_result', 0.0))

        # Construct loss (twoIstar assumed lower is better). Only tau and rho are targeted.
        loss = (
            w['tau'] * _relative_error(target_tau, tau_res) +
            w['rho'] * _relative_error(target_rho, rho_res) +
            w['twoIstar'] * max(twoistar, 0.0)
        )
        if not np.isfinite(loss):
            loss = float('inf')

        info = {
            'tau_res': tau_res,
            'gamma_res': gamma_res,
            'r0_res': r0_res,
            'rho_res': rho_res,
            'twoIstar': twoistar
        }
        return loss, info
    except Exception:
        return float('inf'), {'error': True}


def random_search_hpo(
    wizard,
    bounds: dict,
    n_iter: int = 30,
    seed: int | None = None,
    weights: dict | None = None
) -> tuple[dict, float, dict]:
    """
    Basic random search over provided bounds to find hyperparameters minimizing the loss.

    Parameters
    ----------
    wizard : object
        The MLE wizard instance.
    bounds : dict
        Mapping name -> (low, high, type_str) where type_str in {'int','float'}.
        For paired constraints (e.g., micro_time_start/stop), encode separately and ensure
        low/high reflect valid GUI ranges.
    n_iter : int
        Number of random samples to evaluate.
    seed : int or None
        Random seed for reproducibility.
    weights : dict or None
        Passed to evaluate_hpo_configuration.

    Returns
    -------
    (best_cfg, best_loss, best_info)
    """
    rng = np.random.default_rng(seed)

    def sample_one() -> dict:
        cfg = {}
        for k, (lo, hi, typ) in bounds.items():
            if typ == 'int':
                lo_i = int(np.ceil(lo))
                hi_i = int(np.floor(hi))
                if hi_i <= lo_i:
                    val = lo_i
                else:
                    val = int(rng.integers(lo_i, hi_i + 1))
            else:
                val = float(rng.uniform(float(lo), float(hi)))
            cfg[k] = val
        # enforce start < stop if both present
        if 'micro_time_start' in cfg and 'micro_time_stop' in cfg:
            if int(cfg['micro_time_stop']) <= int(cfg['micro_time_start']):
                cfg['micro_time_stop'] = int(cfg['micro_time_start']) + 1
        if 'irf_start' in cfg and 'irf_stop' in cfg:
            if int(cfg['irf_stop']) <= int(cfg['irf_start']):
                cfg['irf_stop'] = int(cfg['irf_start']) + 1
        return cfg

    best_cfg: dict | None = None
    best_loss: float = float('inf')
    best_info: dict = {}

    for _ in range(max(1, int(n_iter))):
        cfg = sample_one()
        loss, info = evaluate_hpo_configuration(wizard, cfg, weights=weights)
        if loss < best_loss:
            best_loss = loss
            best_cfg = cfg
            best_info = info

    if best_cfg is None:
        best_cfg = {}
        best_loss = float('inf')
        best_info = {'error': True}
    return best_cfg, best_loss, best_info



def optimize_hyperparameters(
        wizard,
        n_iter: int = 40,
        bounds: dict | None = None,
        seed: int | None = None,
        weights: dict | None = None
):
    """
    Externalized HPO routine formerly implemented inside the wizard.
    Keeps identical behavior (progress dialog, cancel handling, best config applied).
    """

    # Ensure we have data to fit
    wizard.update_decay_of_detector()
    if getattr(wizard, 'decay_of_current_file', None) is None:
        QtWidgets.QMessageBox.warning(wizard, "HPO", "No data/decay available. Load bursts and try again.")
        return

    # ---- Prepare sane defaults & merge user bounds ----
    n_bins = max(2, int(len(wizard.irf) // 2))
    mt_start, mt_stop = map(int, wizard.micro_time_range)
    irf_start_cur = int(wizard.irf_start)
    irf_stop_cur = int(wizard.irf_stop)
    shift_cur = int(wizard.shift)

    default_bounds = {
        'irf_threshold_vv': (0.0, 0.5, 'float'),
        'irf_threshold_vh': (0.0, 0.5, 'float'),
        'shift': (shift_cur - 5, shift_cur + 5, 'int'),
        'shift_sp': (-0.9, 0.9, 'float'),
        'shift_ss': (-0.9, 0.9, 'float'),
        'irf_start': (max(0, irf_start_cur - 10), min(n_bins - 2, irf_start_cur + 10), 'int'),
        'irf_stop': (max(1, irf_stop_cur - 10), min(n_bins - 1, irf_stop_cur + 10), 'int'),
        'micro_time_start': (max(0, mt_start - 10), max(0, min(n_bins - 2, mt_start + 10)), 'int'),
        'micro_time_stop': (max(1, mt_stop - 10), min(n_bins - 1, mt_stop + 10), 'int'),
    }
    use_bounds = {**default_bounds, **(bounds or {})}

    # Normalize any inverted / degenerate ranges
    for k, (lo, hi, typ) in list(use_bounds.items()):
        lo = float(lo)
        hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        if typ == 'int':
            lo = int(np.floor(lo))
            hi = int(np.ceil(hi))
            if hi < lo:
                hi = lo
        use_bounds[k] = (lo, hi, typ)

    # ---- Progress dialog ----
    total_budget = int(max(1, n_iter))
    progress = QtWidgets.QProgressDialog("Optimizing hyperparameters...", "Cancel", 0, total_budget, wizard)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setAutoClose(True)
    progress.show()

    # ---- Utilities ----
    rng = np.random.default_rng(seed)
    keys = list(use_bounds.keys())

    def enforce_constraints(cfg: dict) -> dict:
        """Enforce ordering & integer casting."""
        c = dict(cfg)
        # ints
        for k, (lo, hi, typ) in use_bounds.items():
            if typ == 'int':
                c[k] = int(round(c[k]))
            else:
                c[k] = float(c[k])
            # clamp
            c[k] = max(lo, min(hi, c[k]))
        # order constraints
        if 'micro_time_start' in c and 'micro_time_stop' in c:
            if int(c['micro_time_stop']) <= int(c['micro_time_start']):
                c['micro_time_stop'] = int(c['micro_time_start']) + 1
                lo, hi, _ = use_bounds['micro_time_stop']
                c['micro_time_stop'] = int(max(lo, min(hi, c['micro_time_stop'])))
        if 'irf_start' in c and 'irf_stop' in c:
            if int(c['irf_stop']) <= int(c['irf_start']):
                c['irf_stop'] = int(c['irf_start']) + 1
                lo, hi, _ = use_bounds['irf_stop']
                c['irf_stop'] = int(max(lo, min(hi, c['irf_stop'])))
        return c

    def cfg_to_key(c: dict) -> tuple:
        """Hashable key for caching."""
        out = []
        for k in keys:
            v = c[k]
            if use_bounds[k][2] == 'int':
                out.append(int(v))
            else:
                # round floats to avoid tiny duplicates from refinement
                out.append(float(np.round(v, 6)))
        return tuple(out)

    # ---- Latin Hypercube Sampling (exploration) ----
    # Allocate ~70% of budget to LHS, rest to local refinement.
    lhs_budget = max(8, int(0.7 * total_budget))
    ref_budget = max(0, total_budget - lhs_budget)

    D = len(keys)
    # Build LHS in [0,1] for each dimension, then map to bounds
    strata = (np.arange(lhs_budget) + rng.random(lhs_budget)) / lhs_budget  # shape: (lhs_budget,)
    X = np.zeros((lhs_budget, D), dtype=float)
    for j, k in enumerate(keys):
        X[:, j] = strata.copy()
        rng.shuffle(X[:, j])  # independent shuffle per-dimension

    def unscale_row(row01: np.ndarray) -> dict:
        cfg = {}
        for j, k in enumerate(keys):
            lo, hi, typ = use_bounds[k]
            if typ == 'int':
                # inclusive integer mapping
                lo_i, hi_i = int(lo), int(hi)
                if hi_i <= lo_i:
                    cfg[k] = lo_i
                else:
                    # map [0,1) → [lo_i, hi_i] inclusive
                    val = lo_i + int(np.floor(row01[j] * (hi_i - lo_i + 1)))
                    cfg[k] = int(max(lo_i, min(hi_i, val)))
            else:
                cfg[k] = float(lo + row01[j] * (hi - lo))
        return enforce_constraints(cfg)

    # ---- Evaluation with cache + progress ----
    eval_cache: dict[tuple, tuple[float, dict]] = {}
    best_cfg, best_loss, best_info = None, float('inf'), {}

    eval_count = 0

    def evaluate(cfg: dict):
        nonlocal eval_count, best_cfg, best_loss, best_info
        c = enforce_constraints(cfg)
        ck = cfg_to_key(c)
        if ck in eval_cache:
            return eval_cache[ck][0], eval_cache[ck][1], c
        loss, info = evaluate_hpo_configuration(wizard, c, weights=weights)
        eval_cache[ck] = (loss, info)
        eval_count += 1
        if loss < best_loss:
            best_cfg, best_loss, best_info = c, loss, info
        progress.setValue(min(eval_count, total_budget))
        QtWidgets.QApplication.processEvents()
        return loss, info, c

    # LHS pass
    for i in range(lhs_budget):
        if progress.wasCanceled():
            break
        cfg0 = unscale_row(X[i])
        evaluate(cfg0)
        if progress.wasCanceled():
            break

    # ---- Local refinement (coordinate line-search) ----
    # Try to improve around current best with adaptive step sizes.
    if not progress.wasCanceled() and best_cfg is not None and ref_budget > 0:
        # Initial step sizes: 1/4 of range for floats, 1 for ints
        step = {}
        for k, (lo, hi, typ) in use_bounds.items():
            if typ == 'int':
                step[k] = 1
            else:
                step[k] = 0.25 * (hi - lo)

        remain = ref_budget
        no_improve_rounds = 0
        max_no_improve_rounds = 3  # shrink and stop if no progress for a few rounds

        while remain > 0 and not progress.wasCanceled():
            improved = False
            for k in keys:
                if remain <= 0 or progress.wasCanceled():
                    break
                lo, hi, typ = use_bounds[k]
                s = step[k]
                if typ == 'int':
                    candidates = [best_cfg[k] - s, best_cfg[k] + s]
                else:
                    candidates = [best_cfg[k] - s, best_cfg[k] + s]

                for cand in candidates:
                    if remain <= 0 or progress.wasCanceled():
                        break
                    trial = dict(best_cfg)
                    trial[k] = cand
                    trial = enforce_constraints(trial)
                    # If nothing changed after constraints, skip
                    if cfg_to_key(trial) == cfg_to_key(best_cfg):
                        continue
                    loss, _, _ = evaluate(trial)
                    remain -= 1
                    if loss < best_loss:
                        improved = True
                        # update best immediately (evaluate() already does)
                # shrink step if we didn't move on this dim
                if not improved:
                    if typ == 'int':
                        step[k] = max(1, step[k] // 2) if isinstance(step[k], int) else 1
                        # If already 1, keep it; ints are coarse
                    else:
                        step[k] *= 0.5

            if improved:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                # global shrink
                for k in keys:
                    if use_bounds[k][2] == 'int':
                        step[k] = max(1, int(step[k] // 2)) if isinstance(step[k], (int, np.integer)) else 1
                    else:
                        step[k] *= 0.5
            if no_improve_rounds >= max_no_improve_rounds:
                break

    progress.close()

    if best_cfg is None or not np.isfinite(best_loss):
        QtWidgets.QMessageBox.information(wizard, "HPO", "Optimization could not find a valid configuration.")
        return

    # ---- Apply best configuration to UI and refit once ----
    try:
        if 'micro_time_start' in best_cfg and 'micro_time_stop' in best_cfg:
            wizard.micro_time_range = (int(best_cfg['micro_time_start']), int(best_cfg['micro_time_stop']))
            wizard.update_decay_of_detector()
        if 'irf_threshold_vv' in best_cfg:
            wizard.irf_threshold_vv = float(best_cfg['irf_threshold_vv'])
        if 'irf_threshold_vh' in best_cfg:
            wizard.irf_threshold_vh = float(best_cfg['irf_threshold_vh'])
        if 'shift' in best_cfg:
            wizard.shift = int(round(best_cfg['shift']))
        if 'shift_sp' in best_cfg:
            wizard.shift_sp = float(best_cfg['shift_sp'])
        if 'shift_ss' in best_cfg:
            wizard.shift_ss = float(best_cfg['shift_ss'])
        if 'irf_start' in best_cfg:
            wizard.irf_start = int(best_cfg['irf_start'])
        if 'irf_stop' in best_cfg:
            wizard.irf_stop = int(best_cfg['irf_stop'])
    finally:
        wizard._fit = None
        wizard.update_fit()

    # ---- Summary ----
    x0, _ = wizard.fit_parameters
    summary = (
        f"Best loss: {best_loss:.4g}\n"
        f"Targets vs results (tau, rho):\n"
        f"  target: {x0[0]:.4g}, {x0[3]:.4g}\n"
        f"  result: {wizard.tau_result:.4g}, {wizard.rho_result:.4g}\n"
        f"twoI*: {getattr(wizard, 'twoIstar_result', 0.0):.4g}\n\n"
        f"Applied hyperparameters: {best_cfg}\n"
        f"Evaluations: {len(eval_cache)} / budget: {total_budget}"
    )
    QtWidgets.QMessageBox.information(wizard, "HPO complete", summary)
