from __future__ import annotations

from typing import List, Optional
from shlex import split as shlex_split
import numpy as np
from qtpy import QtCore

from .base import BaseCmd

class AnimationMixin(BaseCmd):
    """Timeline control and keyframe animation commands."""

    def _mixin_commands(self):
        return {
            "mset": self._cmd_mset,
            "mdo": self._cmd_mdo,
            "mview": self._cmd_mview,
            "frame": self._cmd_frame,
            "mplay": self._cmd_mplay,
            "mpause": self._cmd_mpause,
            "mstop": self._cmd_mstop,
            "mclear": self._cmd_mclear,
        }

    def _cmd_mset(self, args: List[str]) -> None:
        """Usage: mset specification (e.g. mset 1 x100)"""
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: mset specification")
            return

        spec = " ".join(args).strip()
        # Simple parser for "1 x100" or just "100"
        try:
            if "x" in spec.lower():
                parts = spec.lower().split("x")
                count = int(parts[1].strip())
            else:
                count = int(spec)
            
            viewer.set_total_frames(count)
            self._emit_message(f"Timeline set to {count} frames.")
        except Exception:
            self._emit_error(f"Invalid mset specification: {spec}")

    def _cmd_frame(self, args: List[str]) -> None:
        """Usage: frame index (1-based)"""
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: frame index")
            return

        try:
            val = args[0]
            curr = viewer.get_current_frame()
            total = viewer.get_total_frames()
            
            if val.startswith("+"):
                viewer.set_current_frame((curr + int(val[1:])) % total)
            elif val.startswith("-"):
                if val == "-1": # Last frame
                    viewer.set_current_frame(total - 1)
                else:
                    viewer.set_current_frame((curr - int(val[1:])) % total)
            else:
                frame_no = int(val)
                # Ensure 1-based to 0-based conversion and wrapping
                viewer.set_current_frame((frame_no - 1) % total)
        except Exception:
            self._emit_error("Frame index must be an integer (e.g., 10, +1, -1).")

    def _cmd_mplay(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if viewer._animation_timer is None:
            viewer._animation_timer = QtCore.QTimer(viewer)
            viewer._animation_timer.timeout.connect(self._on_animation_tick)
        
        # Default ~30fps
        viewer._animation_timer.start(33)
        viewer._animation_running = True
        self._emit_message("Playing movie...")

    def _cmd_mpause(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if viewer._animation_timer is not None:
            viewer._animation_timer.stop()
        viewer._animation_running = False
        self._emit_message("Movie paused.")

    def _cmd_mstop(self, args: List[str]) -> None:
        self._cmd_mpause(args)
        window, viewer = self._require_window_and_viewer()
        if viewer is not None:
            viewer.set_current_frame(0)

    def _on_animation_tick(self) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None or not viewer._animation_running:
            return

        curr = viewer.get_current_frame()
        total = viewer.get_total_frames()
        
        next_frame = (curr + 1) % total
        viewer.set_current_frame(next_frame)

    def _cmd_mdo(self, args: List[str]) -> None:
        """Usage: mdo frame, command"""
        self._emit_error("mdo is not yet implemented (deferred to Phase 5.2)")

    def _cmd_mview(self, args: List[str]) -> None:
        """Usage: mview action [, target]"""
        self._emit_error("mview (keyframes) is not yet implemented (deferred to Phase 5.2)")

    def _cmd_mclear(self, args: List[str]) -> None:
        """Clear all animation data."""
        window, viewer = self._require_window_and_viewer()
        if viewer is not None:
            viewer._keyframes.clear()
            viewer.set_total_frames(1)
            self._emit_message("Animation cleared.")
