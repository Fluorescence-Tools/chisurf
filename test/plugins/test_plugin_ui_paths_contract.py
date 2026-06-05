import ast
import pathlib
import re


def _extract_string_constant(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _resolve_decorator_base(py_file: pathlib.Path, path_expr: str, plugins_root: pathlib.Path):
    if not path_expr:
        return py_file.parent
    if "chisurf.core.settings.plugin_path" in path_expr:
        parts = [a or b for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', path_expr)]
        base = plugins_root
        for part in parts:
            base = base / part
        return base
    return None


def test_all_plugin_init_with_ui_paths_exist():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    plugins_root = repo_root / "chisurf" / "plugins"
    missing = []
    unresolved = []

    for py_file in plugins_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fn_name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if fn_name != "init_with_ui":
                continue

            ui_filename = _extract_string_constant(node.args[0]) if node.args else None
            path_expr = ""
            for kw in node.keywords:
                if kw.arg == "ui_filename":
                    ui_filename = _extract_string_constant(kw.value) or ui_filename
                elif kw.arg == "path":
                    path_expr = ast.get_source_segment(source, kw.value) or ""

            if not ui_filename or not ui_filename.endswith(".ui"):
                continue

            base = _resolve_decorator_base(py_file, path_expr, plugins_root)
            if base is None:
                unresolved.append((str(py_file), ui_filename, path_expr))
                continue

            ui_path = base / ui_filename
            if not ui_path.exists():
                missing.append((str(py_file), str(ui_path), path_expr))

    assert not unresolved, f"Unresolved init_with_ui path expressions: {unresolved}"
    assert not missing, f"Missing plugin .ui files: {missing}"


def test_plugin_direct_loadui_string_targets_exist():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    plugins_root = repo_root / "chisurf" / "plugins"
    missing = []

    for py_file in plugins_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fn_name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if fn_name != "loadUi" or not node.args:
                continue

            arg_source = ast.get_source_segment(source, node.args[0]) or ""
            candidates = [a or b for a, b in re.findall(r'"([^"]+\.ui)"|\'([^\']+\.ui)\'', arg_source)]
            for ui_name in candidates:
                ui_path = py_file.parent / ui_name
                if not ui_path.exists():
                    missing.append((str(py_file), ui_name, str(ui_path)))

    assert not missing, f"Missing plugin .ui files in direct loadUi calls: {missing}"
