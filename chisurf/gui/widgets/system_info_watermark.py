from __future__ import annotations

import collections
import os
import sys
import platform
import datetime
import logging
from typing import Deque

from qtpy import QtWidgets, QtCore, QtGui

import chisurf


def _compute_ram_string() -> str:
    """Return total RAM as a human-readable string.

    Tries psutil if available; on Windows falls back to GlobalMemoryStatusEx.
    """
    ram_str = None
    try:  # Prefer psutil if present
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        ram_gb = mem.total / (1024.0 ** 3)
        ram_str = f"{ram_gb:.1f} GB"
    except Exception:
        if os.name == "nt":  # Best-effort Windows fallback
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    ram_gb = stat.ullTotalPhys / (1024.0 ** 3)
                    ram_str = f"{ram_gb:.1f} GB"
            except Exception:
                pass

    if not ram_str:
        ram_str = "N/A"
    return ram_str


def build_system_info_text() -> str:
    """Build a multi-line system info string for the watermark.

    Includes OS, RAM, Python, date, ChiSurf, and selected library versions.
    """
    try:
        os_name = platform.system()
        os_release = platform.release()
        os_version = platform.version()
        platform_info = platform.platform()
        
        # Fix Windows 11 detection and provide detailed info
        if os_name == "Windows":
            if os_release == "11":
                os_info = f"Windows 11 ({os_version})"
            elif os_release == "10" and os_version.startswith("10.0.26"):
                # Windows 11 builds report as Windows 10 10.0.26xxx+
                os_info = f"Windows 11 ({os_version})"
            else:
                os_info = f"Windows {os_release} ({os_version})"
        else:
            os_info = f"{os_name} {os_release}".strip()
    except Exception:
        os_info = "Unknown OS"

    try:
        py_ver = platform.python_version()
    except Exception:
        try:
            py_ver = ".".join(map(str, sys.version_info[:3]))
        except Exception:
            py_ver = "Unknown"

    try:
        from chisurf.core import info as _info_mod  # type: ignore

        cs_ver = getattr(_info_mod, "__version__", None) or getattr(chisurf, "__version__", "?")
    except Exception:
        try:
            cs_ver = getattr(chisurf, "__version__", "?")
        except Exception:
            cs_ver = "?"

    try:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    except Exception:
        date_str = ""

    ram_str = _compute_ram_string()

    def _safe_version(mod_name: str) -> str | None:
        try:
            mod = __import__(mod_name)
        except Exception:
            return None
        for attr in ("__version__", "version", "VERSION"):
            try:
                v = getattr(mod, attr, None)
            except Exception:
                v = None
            if v:
                return str(v)
        return "?"

    numpy_ver = _safe_version("numpy")
    scipy_ver = _safe_version("scipy")
    pandas_ver = _safe_version("pandas")
    chinet_ver = _safe_version("chinet")
    tttrlib_ver = _safe_version("tttrlib")

    try:
        qt_version = getattr(QtCore, "QT_VERSION_STR", None)
    except Exception:
        qt_version = None
    try:
        pyqt_version = getattr(QtCore, "PYQT_VERSION_STR", None)
    except Exception:
        pyqt_version = None
    try:
        import qtpy as _qtpy_mod

        qt_binding = getattr(_qtpy_mod, "QT_API", None)
    except Exception:
        qt_binding = None

    lines = [
        f"OS: {os_info}",
        f"RAM: {ram_str}",
        f"Python: {py_ver}",
    ]
    if date_str:
        lines.append(f"Date: {date_str}")
    lines.append(f"ChiSurf: {cs_ver}")

    if numpy_ver is not None:
        lines.append(f"numpy: {numpy_ver}")
    if scipy_ver is not None:
        lines.append(f"scipy: {scipy_ver}")
    if pandas_ver is not None:
        lines.append(f"pandas: {pandas_ver}")
    if chinet_ver is not None:
        lines.append(f"chinet: {chinet_ver}")
    if tttrlib_ver is not None:
        lines.append(f"tttrlib: {tttrlib_ver}")

    if qt_version or pyqt_version or qt_binding:
        parts = []
        if qt_version:
            parts.append(f"Qt: {qt_version}")
        if pyqt_version:
            parts.append(f"PyQt: {pyqt_version}")
        elif qt_binding:
            parts.append(f"Qt API: {qt_binding}")
        lines.append(" ".join(parts))

    # tttrlib/chinet-related environment variables
    try:
        tttrlib_verbose = os.getenv("TTTRLIB_VERBOSE")
        tttrlib_data = os.getenv("TTTRLIB_DATA")
        tttrlib_use_omp = os.getenv("TTTRLIB_USE_OPENMP")
        tttrlib_num_threads = os.getenv("TTTRLIB_NUM_THREADS")
        chinet_verbose = os.getenv("CHINET_VERBOSE")
    except Exception:
        tttrlib_verbose = tttrlib_data = tttrlib_use_omp = tttrlib_num_threads = chinet_verbose = None

    env_parts: list[str] = []
    if tttrlib_verbose:
        env_parts.append(f"TTTRLIB_VERBOSE={tttrlib_verbose}")
    if tttrlib_data:
        env_parts.append(f"TTTRLIB_DATA={tttrlib_data}")
    if tttrlib_use_omp:
        env_parts.append(f"TTTRLIB_USE_OPENMP={tttrlib_use_omp}")
    if tttrlib_num_threads:
        env_parts.append(f"TTTRLIB_NUM_THREADS={tttrlib_num_threads}")
    if chinet_verbose:
        env_parts.append(f"CHINET_VERBOSE={chinet_verbose}")
    if env_parts:
        env_block = "Env(tttr/chinet):\n  " + "\n  ".join(env_parts)
        lines.append(env_block)

    # Thread-related environment variables used for heavy libs
    thread_parts: list[str] = []
    numba_threads = os.getenv("NUMBA_NUM_THREADS")
    mkl_threads = os.getenv("MKL_NUM_THREADS")
    omp_threads = os.getenv("OMP_NUM_THREADS")
    mkl_layer = os.getenv("MKL_THREADING_LAYER")
    if numba_threads:
        thread_parts.append(f"NUMBA={numba_threads}")
    if mkl_threads:
        thread_parts.append(f"MKL={mkl_threads}")
    if omp_threads:
        thread_parts.append(f"OMP={omp_threads}")
    if mkl_layer:
        thread_parts.append(f"MKL_LAYER={mkl_layer}")
    if thread_parts:
        thread_block = "Threads:\n  " + "\n  ".join(thread_parts)
        lines.append(thread_block)

    # Custom environment variables injected from settings (env)
    try:
        cs_cfg = getattr(chisurf.core.settings, "cs_settings", {})  # type: ignore[attr-defined]
        env_cfg = cs_cfg.get("env", {}) if isinstance(cs_cfg, dict) else {}
        if isinstance(env_cfg, dict) and env_cfg:
            injected_parts: list[str] = []
            for key, configured_val in env_cfg.items():
                # Skip control flag if present
                if key in ("env_override_existing", None):
                    continue
                try:
                    key_str = str(key)
                except Exception:
                    continue
                current_val = os.getenv(key_str)
                display_val = current_val if current_val is not None else "<unset>"
                try:
                    configured_str = str(configured_val)
                except Exception:
                    configured_str = "<unreadable>"
                if current_val is None:
                    injected_parts.append(f"{key_str}={display_val} (configured {configured_str})")
                elif configured_str and current_val != configured_str:
                    injected_parts.append(f"{key_str}={display_val} (configured {configured_str})")
                else:
                    injected_parts.append(f"{key_str}={display_val}")
            if injected_parts:
                lines.append("Env(custom):\n  " + "\n  ".join(injected_parts))
    except Exception:
        pass

    # Current logging/debug level
    try:
        root_logger = logging.getLogger()
        lvl_num = root_logger.getEffectiveLevel()
        lvl_name = logging.getLevelName(lvl_num)
        lines.append(f"Log level: {lvl_name} ({lvl_num})")
    except Exception:
        pass

    return "\n".join(lines)


def _get_memory_usage_mb() -> float | None:
    """Return current process memory usage in MB (including children), or None if unavailable."""
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info().rss
        # Include child processes
        try:
            for child in process.children(recursive=True):
                try:
                    mem += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        return mem / (1024.0 * 1024.0)
    except Exception:
        return None


def _get_cpu_usage_percent() -> float | None:
    """Return CPU usage percentage for ChiSurf process and children.
    
    Returns combined CPU percent across all cores (0-100 * num_cores).
    """
    try:
        import psutil
        process = psutil.Process()
        # Get CPU percent for main process (non-blocking with interval=None uses cached value)
        cpu = process.cpu_percent(interval=None)
        # Include child processes
        try:
            for child in process.children(recursive=True):
                try:
                    cpu += child.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        return cpu
    except Exception:
        return None


def _get_total_memory_mb() -> float | None:
    """Return total system memory in MB, or None if unavailable."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024.0 * 1024.0)
    except Exception:
        return None


def _get_update_interval_ms() -> int:
    """Get memory update interval from settings (default 5000ms)."""
    try:
        cs_settings = getattr(chisurf.core.settings, "cs_settings", {})
        gui_settings = cs_settings.get("gui", {}) if isinstance(cs_settings, dict) else {}
        return int(gui_settings.get("memory_widget_update_interval_ms", 5000))
    except Exception:
        return 5000


class _Sparkline(QtWidgets.QWidget):
    """A small sparkline widget showing value history as a thin line."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        max_samples: int = 40,
        width: int = 80,
        height: int = 16,
        color: QtGui.QColor | None = None,
        fixed_max: float | None = None,
        tooltip: str = "Usage history",
    ):
        super().__init__(parent)
        self._max_samples = max_samples
        self._width = width
        self._height = height
        self._values: Deque[float] = collections.deque(maxlen=max_samples)
        self._color = color or QtGui.QColor(100, 200, 100)
        self._fixed_max = fixed_max  # If set, use fixed max for scaling

        self.setFixedSize(width, height)
        self.setToolTip(tooltip)

    def set_color(self, color: QtGui.QColor) -> None:
        """Update the line color."""
        self._color = color
        self.update()

    def add_value(self, value: float) -> None:
        """Add a new value to the history."""
        self._values.append(value)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        if len(self._values) < 2:
            painter.end()
            return

        # Dynamic range based on actual values with 10% padding
        min_val = min(self._values)
        max_val = max(self._values)
        
        if self._fixed_max is not None:
            # Use fixed max (e.g., 100% for CPU)
            plot_min = 0
            plot_max = self._fixed_max
        else:
            val_range = max_val - min_val
            if val_range < 1.0:
                val_range = max(1.0, max_val * 0.1)
            padding = val_range * 0.1
            plot_min = max(0, min_val - padding)
            plot_max = max_val + padding
        
        plot_range = plot_max - plot_min

        usable_height = self._height - 4
        usable_width = self._width - 4
        step = usable_width / (self._max_samples - 1)

        # Build path
        path = QtGui.QPainterPath()
        values_list = list(self._values)

        for i, value in enumerate(values_list):
            ratio = (value - plot_min) / plot_range if plot_range > 0 else 0.5
            ratio = max(0, min(1, ratio))  # Clamp to [0, 1]
            x = 2 + i * step
            y = self._height - 2 - int(ratio * usable_height)

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        pen = QtGui.QPen(self._color)
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.drawPath(path)

        painter.end()


class _WatermarkLabel(QtWidgets.QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._full_text = ""
        self._folded = False

    def setText(self, text) -> None:  # type: ignore[override]
        try:
            self._full_text = "" if text is None else str(text)
        except Exception:
            self._full_text = ""
        if self._folded:
            display_text = self._build_folded_text()
        else:
            display_text = self._full_text
        super().setText(display_text)

    def _build_folded_text(self) -> str:
        if not self._full_text:
            return ""
        try:
            first_line = self._full_text.splitlines()[0]
        except Exception:
            first_line = self._full_text
        return first_line

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._folded = not self._folded
            if self._folded:
                text = self._build_folded_text()
            else:
                text = self._full_text
            super().setText(text)
            try:
                update_geometry(self.parent())
            except Exception:
                self.adjustSize()
        except Exception:
            pass
        try:
            super().mouseDoubleClickEvent(event)
        except Exception:
            pass


class _WatermarkWidget(QtWidgets.QWidget):
    """Composite watermark widget with system info text, memory and CPU sparklines."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("widget_system_info_watermark")
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        # CPU usage row (sparkline + label)
        self._cpu_row = QtWidgets.QWidget(self)
        cpu_layout = QtWidgets.QHBoxLayout(self._cpu_row)
        cpu_layout.setContentsMargins(0, 0, 0, 0)
        cpu_layout.setSpacing(4)

        self._cpu_sparkline = _Sparkline(
            self._cpu_row,
            color=QtGui.QColor(100, 180, 220),  # Blue
            fixed_max=100.0,  # CPU percentage 0-100
            tooltip="CPU usage history",
        )
        cpu_layout.addWidget(self._cpu_sparkline)

        self._cpu_label = QtWidgets.QLabel("CPU: --", self._cpu_row)
        self._cpu_label.setStyleSheet(
            "color: rgba(255,255,255,200); font-size: 9px; background: transparent;"
        )
        cpu_layout.addWidget(self._cpu_label)
        cpu_layout.addStretch()

        layout.addWidget(self._cpu_row)

        # Memory usage row (sparkline + label)
        self._mem_row = QtWidgets.QWidget(self)
        mem_layout = QtWidgets.QHBoxLayout(self._mem_row)
        mem_layout.setContentsMargins(0, 0, 0, 0)
        mem_layout.setSpacing(4)

        self._mem_sparkline = _Sparkline(
            self._mem_row,
            color=QtGui.QColor(100, 200, 100),  # Green
            tooltip="Memory usage history",
        )
        mem_layout.addWidget(self._mem_sparkline)

        self._mem_label = QtWidgets.QLabel("Mem: --", self._mem_row)
        self._mem_label.setStyleSheet(
            "color: rgba(255,255,255,200); font-size: 9px; background: transparent;"
        )
        mem_layout.addWidget(self._mem_label)
        mem_layout.addStretch()

        layout.addWidget(self._mem_row)

        # System info label
        self._info_label = _WatermarkLabel(self)
        self._info_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self._info_label.setStyleSheet(
            "color: rgba(255,255,255,190); background: transparent; font-size: 9px;"
        )
        layout.addWidget(self._info_label)

        # Widget styling
        self.setStyleSheet(
            "background-color: rgba(0,0,0,96); border-radius: 3px;"
        )

        # Total memory for percentage calculation
        self._total_mem_mb = _get_total_memory_mb() or 16000.0
        
        # Get CPU count for percentage normalization
        try:
            import psutil
            self._cpu_count = psutil.cpu_count() or 1
        except Exception:
            self._cpu_count = 1

        # Timer for usage updates
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_usage)
        interval = _get_update_interval_ms()
        self._timer.start(interval)

        # Initial update
        QtCore.QTimer.singleShot(100, self._update_usage)

    def set_info_text(self, text: str) -> None:
        """Set the system info text."""
        self._info_label.setText(text)

    def _update_usage(self) -> None:
        """Fetch current CPU and memory usage and update display."""
        # Update CPU
        cpu_pct = _get_cpu_usage_percent()
        if cpu_pct is not None:
            # Normalize to per-core percentage for display
            cpu_per_core = cpu_pct / self._cpu_count if self._cpu_count > 0 else cpu_pct
            self._cpu_sparkline.add_value(cpu_per_core)
            
            # Color based on usage
            if cpu_per_core < 50:
                color = QtGui.QColor(100, 180, 220)  # Blue
            elif cpu_per_core < 80:
                color = QtGui.QColor(220, 180, 50)  # Yellow
            else:
                color = QtGui.QColor(220, 80, 80)  # Red
            self._cpu_sparkline.set_color(color)
            
            self._cpu_label.setText(f"CPU: {cpu_per_core:.0f}%")
        else:
            self._cpu_label.setText("CPU: N/A")

        # Update Memory
        mem_mb = _get_memory_usage_mb()
        if mem_mb is not None:
            self._mem_sparkline.add_value(mem_mb)

            # Color based on memory usage percentage
            pct = (mem_mb / self._total_mem_mb) * 100 if self._total_mem_mb > 0 else 0
            if pct < 50:
                color = QtGui.QColor(100, 200, 100)  # Green
            elif pct < 75:
                color = QtGui.QColor(220, 180, 50)  # Yellow
            else:
                color = QtGui.QColor(220, 80, 80)  # Red
            self._mem_sparkline.set_color(color)

            if mem_mb >= 1024:
                mem_str = f"{mem_mb / 1024:.1f} GB"
            else:
                mem_str = f"{mem_mb:.0f} MB"

            self._mem_label.setText(f"Mem: {mem_str} ({pct:.0f}%)")
        else:
            self._mem_label.setText("Mem: N/A")


def ensure_watermark(
    mdiarea: "QtWidgets.QMdiArea | None", existing_widget: QtWidgets.QWidget | None = None
) -> QtWidgets.QWidget | None:
    """Create or update the watermark widget for the given QMdiArea.

    Returns the widget instance (existing or newly created) or None if mdiarea
    is not available. The widget text and geometry are set.
    """
    if mdiarea is None:
        return existing_widget

    widget = existing_widget
    if widget is None:
        try:
            viewport = mdiarea.viewport()
        except Exception:
            viewport = mdiarea
        widget = _WatermarkWidget(viewport)

    try:
        widget.set_info_text(build_system_info_text())
    except Exception:
        pass

    update_geometry(widget)
    return widget


def update_geometry(widget: QtWidgets.QWidget | None) -> None:
    """Position the watermark widget in the top-right corner of its parent."""
    if widget is None:
        return
    try:
        parent = widget.parent()
        if parent is None:
            return
        rect = parent.rect()
        widget.adjustSize()
        size = widget.sizeHint()
        margin = 10
        x = rect.right() - size.width() - margin
        y = rect.top() + margin

        try:
            mdiarea = parent
            if not hasattr(mdiarea, "subWindowList"):
                try:
                    gp = parent.parent()
                except Exception:
                    gp = None
                if gp is not None and hasattr(gp, "subWindowList"):
                    mdiarea = gp
                else:
                    mdiarea = None

            if mdiarea is not None:
                try:
                    subwindows = list(mdiarea.subWindowList())
                except Exception:
                    subwindows = []

                if subwindows:
                    widget_rect = QtCore.QRect(x, y, size.width(), size.height())

                    def _overlaps_any(r: QtCore.QRect) -> bool:
                        for sw in subwindows:
                            try:
                                if not sw.isVisible():
                                    continue
                                if sw.geometry().intersects(r):
                                    return True
                            except Exception:
                                continue
                        return False

                    if _overlaps_any(widget_rect):
                        positions = [
                            (rect.right() - size.width() - margin, rect.top() + margin),
                            (rect.left() + margin, rect.top() + margin),
                            (rect.right() - size.width() - margin, rect.bottom() - size.height() - margin),
                            (rect.left() + margin, rect.bottom() - size.height() - margin),
                        ]

                        chosen_rect = None
                        for px, py in positions:
                            candidate = QtCore.QRect(px, py, size.width(), size.height())
                            if not _overlaps_any(candidate):
                                chosen_rect = candidate
                                break

                        if chosen_rect is None:
                            widget.hide()
                            return

                        x = chosen_rect.x()
                        y = chosen_rect.y()
        except Exception:
            pass

        widget.setGeometry(x, y, size.width(), size.height())
        widget.lower()
        widget.show()
    except Exception:
        pass
