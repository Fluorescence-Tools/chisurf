from __future__ import annotations

import os
import sys
import platform
import datetime
import logging

from qtpy import QtWidgets, QtCore

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
        from chisurf import info as _info_mod  # type: ignore

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

    # Current logging/debug level
    try:
        root_logger = logging.getLogger()
        lvl_num = root_logger.getEffectiveLevel()
        lvl_name = logging.getLevelName(lvl_num)
        lines.append(f"Log level: {lvl_name} ({lvl_num})")
    except Exception:
        pass

    return "\n".join(lines)


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
                update_geometry(self)
            except Exception:
                self.adjustSize()
        except Exception:
            pass
        try:
            super().mouseDoubleClickEvent(event)
        except Exception:
            pass


def ensure_watermark(
    mdiarea: "QtWidgets.QMdiArea | None", existing_label: QtWidgets.QLabel | None = None
) -> QtWidgets.QLabel | None:
    """Create or update the watermark label for the given QMdiArea.

    Returns the label instance (existing or newly created) or None if mdiarea
    is not available. The label text and geometry are set.
    """
    if mdiarea is None:
        return existing_label

    label = existing_label
    if label is None:
        try:
            viewport = mdiarea.viewport()
        except Exception:
            viewport = mdiarea
        label = _WatermarkLabel(viewport)
        label.setObjectName("label_system_info_watermark")
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        label.setStyleSheet(
            "color: rgba(255,255,255,190);"
            "background-color: rgba(0,0,0,96);"
            "padding: 2px 6px;"
            "font-size: 9px;"
        )

    try:
        label.setText(build_system_info_text())
    except Exception:
        pass

    update_geometry(label)
    return label


def update_geometry(label: QtWidgets.QLabel | None) -> None:
    """Position the watermark label in the top-right corner of its parent."""
    if label is None:
        return
    try:
        parent = label.parent()
        if parent is None:
            return
        rect = parent.rect()
        label.adjustSize()
        size = label.sizeHint()
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
                    label_rect = QtCore.QRect(x, y, size.width(), size.height())

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

                    if _overlaps_any(label_rect):
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
                            label.hide()
                            return

                        x = chosen_rect.x()
                        y = chosen_rect.y()
        except Exception:
            pass

        label.setGeometry(x, y, size.width(), size.height())
        label.lower()
        label.show()
    except Exception:
        pass
