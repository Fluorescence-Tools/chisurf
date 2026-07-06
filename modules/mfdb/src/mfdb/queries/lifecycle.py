"""Lifecycle state-machine access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``mfdb_state_transition`` surface (current state, history, and
validated transitions) as a mixin. Extracted verbatim from the former repository
god-class; behaviour is unchanged. The lifecycle *definitions* live in
``mfdb.lifecycle``; this is the per-entity transition log.
"""

from __future__ import annotations

from typing import Any

from mfdb._sqlutil import _utc_now


class LifecycleMixin:
    """Entity lifecycle state, history, and validated transitions (PRD-12)."""

    def get_state(self, entity_type: str, entity_id: str) -> str | None:
        """Return an entity's current lifecycle state, or ``None``.

        The current state is the latest non-deleted transition's ``to_state`` — the
        transition log is the source of truth (no separate mutable status flag).
        """
        row = self.conn.execute(
            "SELECT to_state FROM mfdb_state_transition "
            "WHERE entity_type = ? AND entity_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at DESC, transition_id DESC LIMIT 1",
            (entity_type, entity_id),
        ).fetchone()
        return row[0] if row else None

    def get_state_history(
        self, entity_type: str, entity_id: str
    ) -> list[dict[str, Any]]:
        """Return an entity's transitions in chronological order (oldest first)."""
        rows = self.conn.execute(
            "SELECT transition_id, entity_type, entity_id, from_state, to_state, "
            "reason, operator_user_id, created_at FROM mfdb_state_transition "
            "WHERE entity_type = ? AND entity_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at, transition_id",
            (entity_type, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def _state_transition_allowed(
        self, entity_type: str, from_state: str | None, to_state: str
    ) -> bool:
        """Is ``(entity_type, from_state, to_state)`` a declared transition rule?

        A NULL ``from_state`` rule declares an allowed initial state.
        """
        row = self.conn.execute(
            "SELECT 1 FROM mfdb_state_transition_rule "
            "WHERE entity_type = ? AND to_state = ? AND deleted_at IS NULL "
            "AND ((from_state IS NULL AND ? IS NULL) OR from_state = ?) LIMIT 1",
            (entity_type, to_state, from_state, from_state),
        ).fetchone()
        return row is not None

    def transition_state(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        *,
        reason: str = "",
        operator_user_id: str | None = None,
    ) -> bool:
        """Move an entity to ``to_state``, recording the transition (PRD-12).

        Validates against ``mfdb_state_transition_rule`` (raises
        :class:`~mfdb.lifecycle.StateTransitionError` on an illegal jump,
        surfaced not swallowed), is an **idempotent no-op** when already in
        ``to_state`` (returns ``False``), and otherwise records a transition row and
        publishes PRD-21's ``state.changed`` event post-commit (best-effort; a
        subscriber can never break the transition). Returns ``True`` when a transition
        was recorded.
        """
        from mfdb.lifecycle import StateTransitionError

        current = self.get_state(entity_type, entity_id)
        if current == to_state:
            return False
        if not self._state_transition_allowed(entity_type, current, to_state):
            raise StateTransitionError(
                f"illegal {entity_type} transition {current!r} -> {to_state!r} "
                "(no matching mfdb_state_transition_rule)"
            )
        now = _utc_now()
        with self._transaction():
            next_id = (
                self.conn.execute(
                    "SELECT COALESCE(MAX(transition_id), 0) FROM mfdb_state_transition"
                ).fetchone()[0]
                + 1
            )
            self.conn.execute(
                "INSERT INTO mfdb_state_transition "
                "(transition_id, entity_type, entity_id, from_state, to_state, reason, "
                "operator_user_id, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    next_id, entity_type, entity_id, current, to_state,
                    reason or None, operator_user_id, now, now, None,
                ),
            )
            self.add_audit_log(
                action="transition",
                target_type=f"state:{entity_type}",
                target_id=entity_id,
                operator_user_id=operator_user_id,
                details={"from": current, "to": to_state, "reason": reason},
            )
        # Post-commit, best-effort event (PRD-21 Task 3); never breaks the transition.
        from mfdb.events import EVENT_STATE_CHANGED, publish

        publish(
            EVENT_STATE_CHANGED,
            entity_type=entity_type,
            entity_id=entity_id,
            from_state=current,
            to_state=to_state,
            reason=reason or "",
            operator_user_id=operator_user_id or "",
        )
        return True
