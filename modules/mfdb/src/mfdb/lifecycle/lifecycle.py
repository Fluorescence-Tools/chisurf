"""Lifecycle state machine definitions + seeding (PRD-12 LIMS P1).

Entity lifecycles (sample / artifact / operation) are authored once in
``data/state_lifecycle_defs.json`` and seeded into two `.dic`-declared, generated
tables (``reconcile_schema`` creates them):

- per-entity-type **state vocabularies** in ``mfdb_vocabulary``
  (``field_name = "state:<entity_type>"``), and
- the allowed **transition rules** in ``mfdb_state_transition_rule``
  (``entity_type, from_state, to_state``; a null ``from_state`` is an allowed
  initial state).

The transition *log* (``mfdb_state_transition``) is the source of truth for an
entity's current state; the repository API that writes it (``transition_state`` /
``get_state`` / ``get_state_history``) is built on these definitions. This module is
the seeding + read-only definition layer (Increment 1); it has no Qt or registration
side effects.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFS_PATH = Path(__file__).parent.parent / "data" / "state_lifecycle_defs.json"


class StateTransitionError(ValueError):
    """An illegal lifecycle state transition (no matching rule) — PRD-12."""

#: ``mfdb_vocabulary.field_name`` for an entity type's states.
def state_field(entity_type: str) -> str:
    """Return the vocabulary ``field_name`` keying an entity type's states."""
    return f"state:{entity_type}"


@dataclass(frozen=True)
class LifecycleDef:
    """The authored lifecycle for one entity type."""

    entity_type: str
    states: tuple[str, ...]
    #: Allowed ``(from_state, to_state)`` transitions; ``from_state`` is ``None``
    #: for an allowed initial state.
    transitions: tuple[tuple[str | None, str], ...]

    @property
    def initial_states(self) -> tuple[str, ...]:
        return tuple(to for frm, to in self.transitions if frm is None)

    def allows(self, from_state: str | None, to_state: str) -> bool:
        return (from_state, to_state) in self.transitions


def load_lifecycle_defs() -> dict[str, LifecycleDef]:
    """Load the authored lifecycle definitions (single source)."""
    data = json.loads(_DEFS_PATH.read_text(encoding="utf-8"))
    defs: dict[str, LifecycleDef] = {}
    for entity_type, spec in data.items():
        if entity_type.startswith("_"):
            continue
        transitions = tuple(
            (frm, to) for frm, to in spec.get("transitions", [])
        )
        defs[entity_type] = LifecycleDef(
            entity_type=entity_type,
            states=tuple(spec.get("states", [])),
            transitions=transitions,
        )
    return defs


def get_lifecycle_def(entity_type: str) -> LifecycleDef | None:
    """Return the authored :class:`LifecycleDef` for ``entity_type``, or ``None``."""
    return load_lifecycle_defs().get(entity_type)


def bootstrap_lifecycle_defs(conn: sqlite3.Connection) -> None:
    """Seed state vocabularies + transition rules from the authored source (idempotent).

    States go into ``mfdb_vocabulary`` (``field_name = "state:<entity_type>"``,
    ``INSERT OR IGNORE`` so user-added states survive). Transition rules replace the
    authored rows per entity type (so the table always reflects the current source);
    ``rule_id`` is assigned explicitly because the dictionary-generated key column is
    not a rowid alias (the ``operation_parameter_def`` convention).
    """
    defs = load_lifecycle_defs()
    try:
        with conn:
            for entity_type, ld in defs.items():
                for state in ld.states:
                    conn.execute(
                        """INSERT OR IGNORE INTO mfdb_vocabulary (
                            field_name, value, display_name, description,
                            is_builtin, is_active
                        ) VALUES (?, ?, ?, ?, 1, 1)""",
                        (
                            state_field(entity_type), state, state,
                            f"Built-in {entity_type} lifecycle state",
                        ),
                    )
                conn.execute(
                    "DELETE FROM mfdb_state_transition_rule WHERE entity_type = ?",
                    (entity_type,),
                )
            next_id = (
                conn.execute(
                    "SELECT COALESCE(MAX(rule_id), 0) FROM mfdb_state_transition_rule"
                ).fetchone()[0]
                + 1
            )
            for entity_type, ld in defs.items():
                for from_state, to_state in ld.transitions:
                    conn.execute(
                        """INSERT INTO mfdb_state_transition_rule
                           (rule_id, entity_type, from_state, to_state)
                           VALUES (?, ?, ?, ?)""",
                        (next_id, entity_type, from_state, to_state),
                    )
                    next_id += 1
    except sqlite3.OperationalError:
        # Tables not present yet (best-effort during early migration).
        logger.debug("lifecycle tables not available; skipped seeding")
