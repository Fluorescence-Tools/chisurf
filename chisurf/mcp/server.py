from __future__ import annotations

import argparse
import ast
import io
import sys
import contextlib
import traceback
import threading
from typing import Any, Dict, List, Optional

try:
    from qtpy import QtCore, QtWidgets
except Exception:
    QtCore = None
    QtWidgets = None


_gui_sync_invoker = None
_gui_sync_invoker_lock = threading.Lock()


if QtCore is not None:
    class _GuiSyncInvoker(QtCore.QObject):
        invokeRequested = QtCore.Signal(object)

        def __init__(self):
            super().__init__()
            self.invokeRequested.connect(self._invoke, QtCore.Qt.BlockingQueuedConnection)

        @QtCore.Slot(object)
        def _invoke(self, callback):
            callback()
else:
    _GuiSyncInvoker = None

def _ensure_fastmcp():
    try:
        from fastmcp import FastMCP
        return FastMCP
    except ImportError as e:
        raise RuntimeError(
            "fastmcp is not installed. Install it to run the ChiSurf MCP server (e.g. `pip install fastmcp` or via conda). "
            f"Import error: {e}"
        ) from e

def _get_chisurf():
    try:
        import chisurf
        return chisurf
    except ImportError as e:
        raise RuntimeError("Could not import chisurf. Please ensure it is installed and in the PYTHONPATH.") from e


def _execute_on_gui_thread_sync(func):
    if func is None:
        return None

    if QtCore is None or QtWidgets is None or _GuiSyncInvoker is None:
        return func()

    app = QtWidgets.QApplication.instance()
    if app is None:
        return func()

    if QtCore.QThread.currentThread() is app.thread():
        return func()

    global _gui_sync_invoker
    with _gui_sync_invoker_lock:
        if _gui_sync_invoker is None:
            invoker = _GuiSyncInvoker()
            invoker.moveToThread(app.thread())
            _gui_sync_invoker = invoker

    out: Dict[str, Any] = {}

    def _callable_wrapper():
        try:
            out["ok"] = True
            out["value"] = func()
        except Exception as e:
            out["ok"] = False
            out["error"] = e
            out["traceback"] = traceback.format_exc()

    _gui_sync_invoker.invokeRequested.emit(_callable_wrapper)
    if not out.get("ok", False):
        err = out.get("error")
        if isinstance(err, Exception):
            raise err
        raise RuntimeError(str(err or "GUI-thread execution failed"))
    return out.get("value")


def resolve_transport_kwargs(
    transport: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str = "/mcp",
) -> Dict[str, Any]:
    t = str(transport or "stdio").strip().lower()
    p = str(path or "/mcp").strip() or "/mcp"
    if not p.startswith("/"):
        p = f"/{p}"

    if t in ("streamable-http", "sse", "http"):
        return {
            "host": str(host or "127.0.0.1"),
            "port": int(port),
            "path": p,
        }
    return {}

def register_core_tools(mcp) -> None:
    @mcp.tool
    def ping() -> Dict[str, Any]:
        """Lightweight connectivity/health check."""
        try:
            chisurf = _get_chisurf()
            has_qapp = False
            if QtWidgets is not None:
                try:
                    has_qapp = QtWidgets.QApplication.instance() is not None
                except Exception:
                    has_qapp = False
            return {
                "ok": True,
                "server": "chisurf.mcp",
                "chisurf_loaded": chisurf is not None,
                "qt_app": bool(has_qapp),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool
    def list_runtime_vars(prefix: str = "") -> Dict[str, Any]:
        """List top-level runtime names on the live chisurf module."""
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            names = sorted([str(k) for k in vars(chisurf).keys()])
            p = str(prefix or "").strip()
            if p:
                names = [n for n in names if n.startswith(p)]
            return {"ok": True, "names": names, "count": int(len(names))}

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            return {"ok": False, "error": str(e), "names": [], "count": 0}

    @mcp.tool
    def get_runtime_var(name: str, repr_limit: int = 2000) -> Dict[str, Any]:
        """Inspect a runtime object via dotted path, e.g. 'cs.current_setup'."""
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            key = str(name or "").strip()
            if not key:
                return {"ok": False, "error": "name is required"}

            current: Any = chisurf
            for part in key.split("."):
                token = part.strip()
                if not token:
                    continue
                if isinstance(current, dict) and token in current:
                    current = current[token]
                    continue
                if hasattr(current, token):
                    current = getattr(current, token)
                    continue
                return {"ok": False, "error": f"path not found at '{token}'"}

            limit = max(64, int(repr_limit or 2000))
            txt = repr(current)
            if len(txt) > limit:
                txt = txt[:limit] + "..."
            return {
                "ok": True,
                "name": key,
                "type": type(current).__name__,
                "value_repr": txt,
                "is_none": current is None,
            }

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool
    def set_runtime_var(name: str, value: str) -> Dict[str, Any]:
        """Set a top-level chisurf module variable using a literal value string."""
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            key = str(name or "").strip()
            if not key:
                return {"ok": False, "error": "name is required"}
            if "." in key:
                return {"ok": False, "error": "only top-level chisurf names are supported"}

            raw = str(value)
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                parsed = raw

            setattr(chisurf, key, parsed)
            return {
                "ok": True,
                "name": key,
                "type": type(parsed).__name__,
                "value_repr": repr(parsed),
            }

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool
    def show_message(text: str, title: str = "ChiSurf MCP", level: str = "info") -> Dict[str, Any]:
        """Show a debug message in the GUI and write it to logs."""
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            msg = str(text or "")
            ttl = str(title or "ChiSurf MCP")
            lvl = str(level or "info").strip().lower()
            if lvl not in ("info", "warning", "error"):
                lvl = "info"

            try:
                if lvl == "warning":
                    chisurf.logging.warning(msg)
                elif lvl == "error":
                    chisurf.logging.error(msg)
                else:
                    chisurf.logging.info(msg)
            except Exception:
                pass

            if QtWidgets is None:
                return {"ok": True, "shown": False, "reason": "qt unavailable"}
            app = QtWidgets.QApplication.instance()
            if app is None:
                return {"ok": True, "shown": False, "reason": "no qapplication"}

            parent = None
            try:
                parent = getattr(chisurf, "cs", None)
            except Exception:
                parent = None

            if lvl == "warning":
                QtWidgets.QMessageBox.warning(parent, ttl, msg)
            elif lvl == "error":
                QtWidgets.QMessageBox.critical(parent, ttl, msg)
            else:
                QtWidgets.QMessageBox.information(parent, ttl, msg)
            return {"ok": True, "shown": True}

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool
    def discover_actions() -> Dict[str, Any]:
        """Discover available ChiSurf actions."""
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            catalog_fn = getattr(chisurf, "action_catalog", None)
            if callable(catalog_fn):
                value = catalog_fn()
                catalog = list(value) if isinstance(value, list) else []
            else:
                catalog = []
            return {"ok": True, "catalog": catalog, "count": int(len(catalog))}

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            return {"ok": False, "error": str(e), "catalog": [], "count": 0}

    @mcp.tool
    def describe_state() -> Dict[str, Any]:
        """Describe the current state of the ChiSurf runtime, including datasets, fits, and parameters."""
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()

            datasets: List[Dict[str, Any]] = []
            for idx, d in enumerate(list(getattr(chisurf, "imported_datasets", []) or [])):
                datasets.append(
                    {
                        "index": int(idx),
                        "name": str(getattr(d, "name", "") or ""),
                        "uid": str(getattr(d, "unique_identifier", "") or ""),
                        "type": type(d).__name__,
                        "experiment": str(getattr(getattr(d, "experiment", None), "name", "") or ""),
                    }
                )

            fits: List[Dict[str, Any]] = []
            for idx, f in enumerate(list(getattr(chisurf, "fits", []) or [])):
                fit_info = {
                    "index": int(idx),
                    "name": str(getattr(f, "name", "") or ""),
                    "uid": str(getattr(f, "unique_identifier", "") or ""),
                    "type": type(f).__name__,
                    "data_set_name": str(getattr(getattr(f, "data", None), "name", "") or ""),
                    "chi2": None,
                }

                try:
                    fit_info["chi2"] = float(getattr(f, "chi2", float("nan")))
                except Exception:
                    fit_info["chi2"] = None

                parameters = {}
                try:
                    if hasattr(f, "params"):
                        params = f.params
                    elif hasattr(f, "parameters"):
                        params = f.parameters
                    else:
                        params = None

                    if params and hasattr(params, "keys"):
                        for pk in params.keys():
                            p = params[pk]
                            parameters[pk] = {
                                "value": getattr(p, "value", None),
                                "fixed": getattr(p, "fixed", None),
                                "lower_bound": getattr(p, "lower_bound", None),
                                "upper_bound": getattr(p, "upper_bound", None),
                                "linked_to": getattr(getattr(p, "linked_to", None), "name", None),
                            }
                    fit_info["parameters"] = parameters
                except Exception:
                    pass

                fits.append(fit_info)

            current_experiment = ""
            current_setup = ""
            try:
                cs = getattr(chisurf, "cs", None)
                current_experiment = str(getattr(cs, "current_experiment", "") or "")
                current_setup = str(getattr(getattr(cs, "current_setup", None), "name", "") or "")
            except Exception:
                pass

            return {
                "ok": True,
                "state": {
                    "current_experiment": current_experiment,
                    "current_setup": current_setup,
                    "datasets": datasets,
                    "fits": fits,
                },
            }

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool
    def get_fit_quality(fit_index: Optional[int] = None) -> Dict[str, Any]:
        """Return fit quality metrics (chi2 and context) for LLM decision making."""

        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            fits = list(getattr(chisurf, "fits", []) or [])
            if not fits:
                return {"ok": False, "error": "no fits available"}

            if fit_index is None:
                fit = getattr(getattr(chisurf, "cs", None), "current_fit", None)
                if fit is None:
                    fit = fits[0]
                try:
                    idx = int(fits.index(fit))
                except Exception:
                    idx = 0
                    fit = fits[0]
            else:
                idx = int(fit_index)
                if idx < 0 or idx >= len(fits):
                    return {"ok": False, "error": f"fit_index out of range: {idx}"}
                fit = fits[idx]

            chi2 = None
            try:
                chi2 = float(getattr(fit, "chi2", float("nan")))
            except Exception:
                chi2 = None

            return {
                "ok": True,
                "fit_index": int(idx),
                "fit_name": str(getattr(fit, "name", "") or ""),
                "data_set_name": str(getattr(getattr(fit, "data", None), "name", "") or ""),
                "chi2": chi2,
            }

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool
    def run_fit_with_quality(
        fit_index: Optional[int] = None,
        retries: int = 0,
        improvement_epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        """Run a fit and report chi2 before/after (with optional retries)."""

        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            fits = list(getattr(chisurf, "fits", []) or [])
            if not fits:
                return {"ok": False, "error": "no fits available"}

            if fit_index is None:
                fit = getattr(getattr(chisurf, "cs", None), "current_fit", None)
                if fit is None:
                    fit = fits[0]
                try:
                    idx = int(fits.index(fit))
                except Exception:
                    idx = 0
                    fit = fits[0]
            else:
                idx = int(fit_index)
                if idx < 0 or idx >= len(fits):
                    return {"ok": False, "error": f"fit_index out of range: {idx}"}
                fit = fits[idx]

            max_runs = max(1, int(retries) + 1)
            eps = float(improvement_epsilon or 0.0)

            def _safe_chi2() -> Optional[float]:
                try:
                    return float(getattr(fit, "chi2", float("nan")))
                except Exception:
                    return None

            before = _safe_chi2()
            best = before
            run_results: List[Dict[str, Any]] = []

            for i in range(max_runs):
                fit.run()
                after_i = _safe_chi2()
                improved = False
                if best is not None and after_i is not None:
                    improved = (best - after_i) > eps
                elif best is None and after_i is not None:
                    improved = True

                if improved:
                    best = after_i

                run_results.append(
                    {
                        "run": int(i + 1),
                        "chi2": after_i,
                        "improved_vs_best": bool(improved),
                    }
                )

            after = _safe_chi2()
            net_improved = False
            if before is not None and after is not None:
                net_improved = (before - after) > eps

            return {
                "ok": True,
                "fit_index": int(idx),
                "fit_name": str(getattr(fit, "name", "") or ""),
                "chi2_before": before,
                "chi2_after": after,
                "improved": bool(net_improved),
                "runs": run_results,
            }

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            tb = traceback.format_exc()
            return {"ok": False, "error": str(e), "traceback": tb}

    @mcp.tool
    def execute_action(action: str, payload: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a single ChiSurf action through the action controller."""
        action = str(action or "").strip()
        payload = dict(payload or {})
        context = dict(context or {})
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()
            controller = getattr(chisurf, "action_controller", None)
            if controller is None or not hasattr(controller, "execute"):
                execute_fn = getattr(chisurf, "action_execute", None)
                if callable(execute_fn):
                    out = execute_fn(name=action, payload=payload)
                    return {"ok": True, "result": out}
                return {"ok": False, "error": "action controller unavailable"}

            out = controller.execute(name=action, payload=payload, context=context)
            return {"ok": True, "result": out}

        try:
            return _execute_on_gui_thread_sync(_impl)
        except Exception as e:
            tb = traceback.format_exc()
            return {"ok": False, "error": str(e), "traceback": tb}

    @mcp.tool
    def execute_plan(steps: Optional[List[Dict[str, Any]]] = None, stop_on_error: bool = True) -> Dict[str, Any]:
        """Execute a sequence of ChiSurf actions."""
        steps = list(steps or [])
        stop_on_error = bool(stop_on_error)
        
        out_steps: List[Dict[str, Any]] = []
        fail_error = ""
        
        for idx, s in enumerate(steps):
            if not isinstance(s, dict):
                entry = {"index": int(idx), "ok": False, "error": "step must be an object"}
                out_steps.append(entry)
                fail_error = entry["error"]
                if stop_on_error:
                    break
                continue
                
            action_name = str(s.get("action", "") or s.get("name", ""))
            payload = dict(s.get("payload", {}) or {})
            context = dict(s.get("context", {}) or {})
            
            r = execute_action(action=action_name, payload=payload, context=context)
            ok = bool((r or {}).get("ok", False))
            
            out_steps.append({
                "index": int(idx), 
                "ok": ok, 
                "action": action_name, 
                "response": r
            })
            
            if not ok:
                fail_error = str((r or {}).get("error", "step failed") or "step failed")
                if stop_on_error:
                    break
                    
        return {
            "ok": fail_error == "",
            "error": fail_error,
            "results": out_steps,
            "count": int(len(out_steps)),
        }

    @mcp.tool
    def run_script(script_code: str) -> Dict[str, Any]:
        """
        Execute arbitrary Python script code against the live ChiSurf runtime. 
        Provides direct access to the `chisurf` module in the execution locals.
        Returns the captured stdout, stderr, and execution status.
        Use this for complex logic like batch file loading, cross-dataset parameter linking, 
        or other tasks that standard actions cannot easily handle.
        """
        def _impl() -> Dict[str, Any]:
            chisurf = _get_chisurf()

            exec_locals = {
                "chisurf": chisurf,
            }

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            success = False
            error_msg = ""
            traceback_str = ""

            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                try:
                    exec(script_code, exec_locals)
                    success = True
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    traceback_str = traceback.format_exc()

            return {
                "ok": success,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": error_msg,
                "traceback": traceback_str,
            }

        return _execute_on_gui_thread_sync(_impl)

def create_mcp(name: str = "ChiSurf"):
    FastMCP = _ensure_fastmcp()
    mcp = FastMCP(str(name))
    register_core_tools(mcp)
    return mcp


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ChiSurf MCP server")
    parser.add_argument("--name", default="ChiSurf")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument("--show-banner", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    mcp = create_mcp(name=str(args.name))
    kwargs = resolve_transport_kwargs(
        transport=str(args.transport),
        host=str(args.host),
        port=int(args.port),
        path=str(args.path),
    )
    mcp.run(transport=str(args.transport), show_banner=bool(args.show_banner), **kwargs)

if __name__ == "__main__":
    main()
