import ast
import ctypes
import os
import pathlib
import sys

from chisurf.core.plugin.manifest import load_manifest

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
        from qtpy.QtCore import QSize  # type: ignore
        from qtpy.QtGui import QImage, QPainter  # type: ignore
        from qtpy.QtSvg import QSvgRenderer  # type: ignore

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

    def _string_literal(value):
        """Return a string literal value from modern or legacy AST nodes."""
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        ast_str = getattr(ast, "Str", None)
        if ast_str is not None and isinstance(value, ast_str):
            return value.s
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in getattr(node, "targets", []):
            if isinstance(target, ast.Name) and target.id == "name":
                literal = _string_literal(node.value)
                if literal is not None:
                    plugin_name = literal
            if isinstance(target, ast.Name) and target.id == "cli_entrypoint":
                literal = _string_literal(node.value)
                if literal is not None:
                    cli_entrypoint = literal.strip()
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


def _read_manifest_metadata(plugin_dir: pathlib.Path):
    """Read plugin metadata from ``manifest.json`` when present."""
    manifest = load_manifest(plugin_dir / "manifest.json")
    if manifest is None:
        return None

    (
        _legacy_name,
        legacy_description,
        legacy_cli_entrypoint,
        _legacy_cli_only,
        _legacy_menu_hidden,
    ) = _read_plugin_metadata(plugin_dir / "__init__.py")
    cli_entrypoint = manifest.entrypoints.cli or legacy_cli_entrypoint

    return {
        "plugin_name": manifest.display_name or manifest.id,
        "description": manifest.description or legacy_description or "No description available.",
        "cli_entrypoint": cli_entrypoint,
        "cli_only": bool(not manifest.entrypoints.gui),
        "menu_hidden": bool(manifest.menu_hidden),
        "manifest_id": manifest.id,
        "manifest_version": manifest.version,
        "state_namespace": manifest.state_namespace,
    }


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
                manifest_metadata = _read_manifest_metadata(package_dir)
                if manifest_metadata is not None:
                    plugin_name = manifest_metadata["plugin_name"]
                    description = manifest_metadata["description"]
                    cli_entrypoint = manifest_metadata["cli_entrypoint"]
                    cli_only = manifest_metadata["cli_only"]
                    menu_hidden = manifest_metadata["menu_hidden"]
                    manifest_id = manifest_metadata["manifest_id"]
                    manifest_version = manifest_metadata["manifest_version"]
                    state_namespace = manifest_metadata["state_namespace"]
                else:
                    (
                        plugin_name,
                        description,
                        cli_entrypoint,
                        cli_only,
                        menu_hidden,
                    ) = _read_plugin_metadata(init_py)
                    manifest_id = None
                    manifest_version = None
                    state_namespace = None
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
                    "manifest_id": manifest_id,
                    "manifest_version": manifest_version,
                    "state_namespace": state_namespace,
                }
            except Exception:
                continue


class OptionalModuleProxy:
    """A proxy object that behaves as a falsy module and returns itself for any attribute access."""
    def __init__(self, name):
        self.__name__ = name
        self.__path__ = []

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return OptionalModuleProxy(f"{self.__name__}.{name}")

    def __call__(self, *args, **kwargs):
        return None

    def __bool__(self):
        return False


class DevPluginFinder:
    """A MetaPathFinder that provides virtual modules for missing chisurf.plugins._dev subpackages."""
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("chisurf.plugins._dev"):
            try:
                # Check if the physical _dev directory exists.
                # If it exists, we let the normal import system handle it.
                plugins_dir = pathlib.Path(__file__).parent
                if (plugins_dir / "_dev").is_dir():
                    return None
            except Exception:
                pass

            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return OptionalModuleProxy(spec.name)

    def exec_module(self, module):
        pass

# Register the virtual plugin finder
sys.meta_path.append(DevPluginFinder())
