import os
import sys
import pathlib
import ctypes

# Helper to set hidden attribute on Windows

def _set_hidden_on_windows(path: pathlib.Path) -> None:
    if os.name != 'nt':
        return
    try:
        FILE_ATTRIBUTE_HIDDEN = 0x2
        GetFileAttributesW = ctypes.windll.kernel32.GetFileAttributesW
        SetFileAttributesW = ctypes.windll.kernel32.SetFileAttributesW
        GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
        GetFileAttributesW.restype = ctypes.c_uint32
        SetFileAttributesW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
        SetFileAttributesW.restype = ctypes.c_int
        attrs = GetFileAttributesW(str(path))
        if attrs == 0xFFFFFFFF:
            return
        SetFileAttributesW(str(path), attrs | FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        pass

# Define the user plugins directory
user_plugins_dir = pathlib.Path.home() / '.chisurf' / 'plugins'
chisurf_user_dir = user_plugins_dir.parent

# Ensure the base ~/.chisurf exists and is hidden if newly created
base_existed = chisurf_user_dir.exists()
chisurf_user_dir.mkdir(parents=True, exist_ok=True)
if not base_existed:
    _set_hidden_on_windows(chisurf_user_dir)

# Ensure the plugins directory exists
user_plugins_dir.mkdir(parents=True, exist_ok=True)

# Add the user plugins directory to the module's __path__
if str(user_plugins_dir) not in __path__:
    __path__.append(str(user_plugins_dir))


# --- Auto-rasterize plugin SVG icons to PNG (once per process) ---
_def_done_flag = '_chisurf_plugins_svg_rasterized'
if not getattr(sys.modules.get(__name__), _def_done_flag, False):
    setattr(sys.modules.get(__name__), _def_done_flag, True)
    try:
        # Import QtSvg lazily to avoid hard dependency if GUI isn't used
        from PyQt5.QtSvg import QSvgRenderer  # type: ignore
        from PyQt5.QtGui import QImage, QPainter  # type: ignore
        from PyQt5.QtCore import QSize  # type: ignore
        # If Qt isn't set up (e.g., headless), this block may still work since QImage is offscreen.
        def _rasterize_svg_to_png(svg_path: pathlib.Path, png_path: pathlib.Path, size: QSize = None) -> bool:
            try:
                renderer = QSvgRenderer(str(svg_path))
                if not renderer.isValid():
                    return False
                default_size = renderer.defaultSize()
                if size is None:
                    # Use SVG default size if available, else fall back to 128x128
                    if default_size.width() > 0 and default_size.height() > 0:
                        size = default_size
                    else:
                        size = QSize(128, 128)
                img = QImage(size, QImage.Format_ARGB32_Premultiplied)
                img.fill(0x00000000)
                painter = QPainter(img)
                try:
                    renderer.render(painter)
                finally:
                    painter.end()
                # Ensure parent dir exists
                png_path.parent.mkdir(parents=True, exist_ok=True)
                return img.save(str(png_path))
            except Exception:
                return False

        # Scan installed package plugins directory for icon.svg files
        pkg_plugins_dir = pathlib.Path(__file__).parent
        for d in pkg_plugins_dir.iterdir():
            try:
                if not d.is_dir():
                    continue
                svg = d / 'icon.svg'
                png = d / 'icon.png'
                if svg.exists() and not png.exists():
                    _rasterize_svg_to_png(svg, png)
            except Exception:
                # Never fail import due to icon generation issues
                pass
    except Exception:
        # QtSvg not available or other import issue; skip rasterization silently
        pass
