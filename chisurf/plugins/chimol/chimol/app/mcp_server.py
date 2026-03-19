from __future__ import annotations
import sys
import traceback
from typing import Optional, Dict, Any, List

try:
    from qtpy import QtCore, QtWidgets
except Exception:
    QtCore = None
    QtWidgets = None

def _get_fastmcp():
    try:
        from fastmcp import FastMCP
        return FastMCP
    except ImportError:
        return None

def _execute_on_gui(func):
    if func is None: return None
    if QtCore is None or QtWidgets is None: return func()
    app = QtWidgets.QApplication.instance()
    if app is None: return func()
    if QtCore.QThread.currentThread() is app.thread(): return func()

    # Simple invoker for ChiMol specifically
    class Invoker(QtCore.QObject):
        req = QtCore.Signal(object)
        def __init__(self):
            super().__init__()
            self.req.connect(self._exec, QtCore.Qt.BlockingQueuedConnection)
        def _exec(self, cb): cb()

    invoker = Invoker()
    invoker.moveToThread(app.thread())
    out = {}
    def wrapper():
        try:
            out["ok"] = True
            out["v"] = func()
        except Exception as e:
            out["ok"] = False
            out["e"] = str(e)
            out["tb"] = traceback.format_exc()
    invoker.req.emit(wrapper)
    if not out.get("ok"):
        raise RuntimeError(f"GUI Error: {out.get('e')}\n{out.get('tb')}")
    return out.get("v")

def create_mcp(window=None):
    FastMCP = _get_fastmcp()
    if not FastMCP:
        print("[Warning] fastmcp not installed. MCP server disabled.", file=sys.stderr)
        return None

    mcp = FastMCP("ChiMol")

    # If window is None, we'll try to find the active ChiMol window via singleton or global
    def get_cmd():
        # This is a bit of a hack, but ChiMol often registers its cmd in its package
        from ..cmd import cmd
        if window:
            cmd.set_window(window)
        return cmd

    @mcp.tool
    def run_command(line: str) -> str:
        """Execute a ChiMol/PyMOL command (e.g. 'load 1d3z; show sticks')."""
        def _impl():
            cmd = get_cmd()
            # Capture messages if possible, but for now just execute
            cmd.do(line)
            return f"Executed: {line}"
        return _execute_on_gui(_impl)

    @mcp.tool
    def list_objects() -> List[Dict[str, Any]]:
        """List all loaded molecular objects and their visibility."""
        def _impl():
            cmd = get_cmd()
            win = cmd.window
            if not win or not hasattr(win, "viewer"):
                return []
            return win.viewer.list_objects()
        return _execute_on_gui(_impl)

    @mcp.tool
    def get_object_info(object_id: str) -> Dict[str, Any]:
        """Get detailed info about a specific object (residue count, etc)."""
        def _impl():
            cmd = get_cmd()
            win = cmd.window
            if not win or not hasattr(win, "viewer"):
                return {}
            seq, names = win.viewer.get_sequence_arrays(object_id)
            return {
                "id": object_id,
                "residue_count": len(seq) if seq is not None else 0,
                "active": win.viewer.get_active_object_id() == object_id
            }
        return _execute_on_gui(_impl)

    return mcp

def main():
    mcp = create_mcp()
    if mcp:
        mcp.run()

if __name__ == "__main__":
    main()
