import os
import sys
import pathlib
import ctypes
import pkgutil
import ast

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


# --- Auto-rasterize plugin SVG icons to PNG (optional) ---
_def_done_flag = '_chisurf_plugins_svg_rasterized'
if (
    os.environ.get("CHISURF_ENABLE_PLUGIN_ICON_RASTERIZE", "").lower()
    in {"1", "true", "yes"}
    and not getattr(sys.modules.get(__name__), _def_done_flag, False)
):
    setattr(sys.modules.get(__name__), _def_done_flag, True)
    try:
        # Import QtSvg lazily to avoid hard dependency if GUI isn't used
        from qtpy.QtSvg import QSvgRenderer  # type: ignore
        from qtpy.QtGui import QImage, QPainter  # type: ignore
        from qtpy.QtCore import QSize  # type: ignore

        def _rasterize_svg_to_png(
            svg_path: pathlib.Path, png_path: pathlib.Path, size: QSize = None
        ) -> bool:
            try:
                renderer = QSvgRenderer(str(svg_path))
                if not renderer.isValid():
                    return False
                default_size = renderer.defaultSize()
                if size is None:
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
                png_path.parent.mkdir(parents=True, exist_ok=True)
                return img.save(str(png_path))
            except Exception:
                return False

        pkg_plugins_dir = pathlib.Path(__file__).parent
        for d in pkg_plugins_dir.iterdir():
            try:
                if not d.is_dir():
                    continue
                svg = d / "icon.svg"
                png = d / "icon.png"
                if svg.exists() and not png.exists():
                    _rasterize_svg_to_png(svg, png)
            except Exception:
                pass
    except Exception:
        pass


def _read_plugin_metadata(init_py: pathlib.Path):
    if not init_py.exists():
        return None, None, None, False, False
    try:
        source = init_py.read_text(encoding="utf-8")
    except Exception:
        return None, None, None
    try:
        tree = ast.parse(source, filename=str(init_py))
    except Exception:
        return None, None, None, False, False
    description = ast.get_docstring(tree) or "No description available."
    plugin_name = None
    cli_entrypoint = None
    cli_only = False
    menu_hidden = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in getattr(node, "targets", []):
            if isinstance(target, ast.Name) and target.id == "name":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    plugin_name = value.value
                elif isinstance(value, ast.Str):
                    plugin_name = value.s
            if isinstance(target, ast.Name) and target.id == "cli_entrypoint":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    cli_entrypoint = value.value.strip()
                elif isinstance(value, ast.Str):
                    cli_entrypoint = value.s.strip()
            if isinstance(target, ast.Name) and target.id == "cli_only":
                value = node.value
                # Support simple boolean literals like ``cli_only = True``.
                if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                    cli_only = bool(value.value)
                else:
                    # Fallback for older Python AST nodes
                    if hasattr(ast, "NameConstant") and isinstance(value, ast.NameConstant):  # type: ignore[attr-defined]
                        if isinstance(value.value, bool):
                            cli_only = bool(value.value)
            if isinstance(target, ast.Name) and target.id == "menu_hidden":
                value = node.value
                # Support simple boolean literals like ``menu_hidden = True``.
                if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                    menu_hidden = bool(value.value)
                else:
                    # Fallback for older Python AST nodes
                    if hasattr(ast, "NameConstant") and isinstance(value, ast.NameConstant):  # type: ignore[attr-defined]
                        if isinstance(value.value, bool):
                            menu_hidden = bool(value.value)
    return plugin_name, description, cli_entrypoint, cli_only, menu_hidden


def iter_plugins():
    base_prefix = __name__ + "."
    try:
        user_root = user_plugins_dir.resolve()
    except Exception:
        user_root = user_plugins_dir
    seen = set()
    search_paths = list(__path__)
    for path_entry in search_paths:
        try:
            base_path = pathlib.Path(path_entry)
        except Exception:
            continue
        try:
            base_path = base_path.resolve()
        except Exception:
            pass
        try:
            if not base_path.exists() or not base_path.is_dir():
                continue
        except Exception:
            continue

        for root, dirs, files in os.walk(str(base_path)):
            try:
                if "__init__.py" not in files:
                    continue
                package_dir = pathlib.Path(root)

                # For nested packages, finder.path already points at the parent
                # directory of the *first* package component. Joining all "parts"
                # would therefore duplicate path segments (e.g. traj/traj_align
                # under a finder.path of .../plugins/traj). Instead, only join the
                # final component relative to finder.path.
                try:
                    rel = package_dir.relative_to(base_path)
                except Exception:
                    continue
                parts = rel.parts
                if not parts:
                    continue

                local_name = parts[-1]
                init_py = package_dir.joinpath("__init__.py")
                if not init_py.exists():
                    continue
                plugin_name, description, cli_entrypoint, cli_only, menu_hidden = _read_plugin_metadata(init_py)
                if not plugin_name:
                    continue
                module_path = base_prefix + ".".join(parts)
                key = (module_path, str(package_dir))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    if hasattr(package_dir, "is_relative_to"):
                        is_user = package_dir.is_relative_to(user_root)
                    else:
                        is_user = str(package_dir).startswith(str(user_root))
                except Exception:
                    is_user = str(package_dir).startswith(str(user_root))

                # Automatically hide the built-in cookiecutter template from the GUI menu
                # while still allowing it to be managed as a plugin if needed.
                if "cookiecutter-chisurf-plugin" in parts and "{{cookiecutter.plugin_name}}" in parts:
                    menu_hidden = True
                yield {
                    "module_path": module_path,
                    "module_name": local_name,
                    "package_dir": package_dir,
                    "source": "user" if is_user else "built-in",
                    "plugin_name": plugin_name,
                    "description": description,
                    "cli_entrypoint": cli_entrypoint,
                    "cli_only": bool(cli_only),
                    "menu_hidden": bool(menu_hidden),
                }
            except Exception:
                continue
