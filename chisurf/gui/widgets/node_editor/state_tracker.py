from __future__ import annotations

from typing import Any, Dict

from qtpy import QtCore, QtWidgets

from .scene import NodeScene

import logging

logger = logging.getLogger(__name__)


class _GraphStateCommand(QtWidgets.QUndoCommand):
    """QUndoCommand storing full graph snapshots before/after a change.

    This is internal to the state tracker module so that we avoid circular
    imports between ``editor.py`` and the tracker implementation.
    """

    def __init__(
        self,
        editor: QtWidgets.QWidget,
        before: Dict[str, Any],
        after: Dict[str, Any],
        *,
        text: str = "",
    ) -> None:
        super().__init__(text or "Graph change")
        self._editor = editor
        self._before = before
        self._after = after
        # First redo() call is invoked immediately when pushing onto the
        # undo stack; skip that to avoid reapplying the state we just set.
        self._applied_once = False

    def _apply(self, state: Dict[str, Any]) -> None:
        try:
            tracker = getattr(self._editor, "state_tracker", None)
            if tracker is not None:
                tracker.apply_state(state)  # type: ignore[arg-type]
        except Exception:
            logger.error("Error applying graph state from _GraphStateCommand", exc_info=True)

    def undo(self) -> None:  # type: ignore[override]
        self._apply(self._before)

    def redo(self) -> None:  # type: ignore[override]
        if not self._applied_once:
            self._applied_once = True
            return
        self._apply(self._after)


class SceneStateTracker(QtCore.QObject):
    """Central helper that captures and reapplies full scene snapshots.

    All undo/redo graph changes flow through this tracker so that we have
    a single source of truth for how scene state is recorded and restored.
    """

    def __init__(
        self,
        editor: QtWidgets.QWidget,
        scene: NodeScene,
        undo_stack: QtWidgets.QUndoStack,
    ) -> None:
        super().__init__(editor)
        self._editor = editor
        self._scene = scene
        self._undo_stack = undo_stack
        self._pending_before: Dict[str, Any] | None = None
        self._pending_text: str = ""
        self._suppress: bool = False

    @property
    def suppress(self) -> bool:
        return self._suppress

    def apply_state(self, data: Dict[str, Any]) -> None:
        """Apply a full graph state to the scene, suppressing undo hooks."""

        if not isinstance(data, dict):
            return
        logger.debug(
            "SceneStateTracker.apply_state: nodes=%d edges=%d",
            len(data.get("nodes", [])),
            len(data.get("edges", [])),
        )
        self._suppress = True
        try:
            self._scene.from_dict(data)
        finally:
            self._suppress = False
        # Ensure any attached views fully repaint so embedded widgets redraw.
        try:
            for view in self._scene.views():
                view.viewport().update()
        except Exception:
            pass

    def begin_action(self, text: str) -> None:
        """Mark the beginning of an undoable scene change."""

        if self._suppress:
            return
        try:
            logger.debug("SceneStateTracker.begin_action text=%s", text)
            self._pending_before = self._scene.to_dict()
            self._pending_text = str(text) if text else "Graph change"
        except Exception as exc:
            logger.error(
                "SceneStateTracker.begin_action: failed to snapshot scene: %s",
                exc,
                exc_info=True,
            )
            self._pending_before = None
            self._pending_text = ""

    def commit_action(self) -> None:
        """Commit the pending snapshot as an undoable command, if changed."""

        logger.debug(
            "SceneStateTracker.commit_action called: suppress=%s pending_before=%s",
            self._suppress,
            self._pending_before is not None,
        )
        if self._suppress:
            logger.debug("SceneStateTracker.commit_action: suppressed; clearing pending state")
            self._pending_before = None
            self._pending_text = ""
            return
        if self._pending_before is None:
            logger.debug("SceneStateTracker.commit_action: no pending snapshot; nothing to do")
            return
        try:
            before = self._pending_before
            after = self._scene.to_dict()
        except Exception:
            self._pending_before = None
            self._pending_text = ""
            return

        self._pending_before = None
        text = self._pending_text or "Graph change"
        self._pending_text = ""
        logger.debug(
            "SceneStateTracker.commit_action: pushing command text=%s nodes_before=%d nodes_after=%d",
            text,
            len(before.get("nodes", [])),
            len(after.get("nodes", [])),
        )

        cmd = _GraphStateCommand(self._editor, before, after, text=text)
        try:
            self._undo_stack.push(cmd)
        except Exception as exc:
            logger.error("Error pushing undo command: %s", exc)
