"""Provenance / lineage query API (PRD-21 Task 1).

A first-class read service over the operation graph so call sites stop hand-writing
`mfdb_operation_artifact`/`mfdb_edge` SQL for "what produced this", "what derives from
this", and "what used this".

The authoritative source is `mfdb_operation_artifact` (the PRD-11 operation node's
typed input/output ports): an artifact is *produced by* the operations that list it as
an ``output`` and *consumed by* those that list it as an ``input``. Ancestry/descent
is the transitive closure over artifact → operation → artifact hops, which is
unambiguous and direction-correct (unlike the dual-written `derived_from`
`mfdb_edge` rows, which this service does not depend on).

>>> from mfdb.lineage import Lineage
>>> lin = Lineage.from_db(db)
>>> lin.ancestors(burst_artifact_id)      # [shifted, raw, …] derivation order
>>> lin.descendants(raw_artifact_id)       # everything derived from the raw file
>>> lin.provenance_graph(artifact_id)      # {"nodes": [...], "edges": [...]}
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

#: ``mfdb_edge`` relationship types meaning "the *source* node was produced or
#: derived **using** the *target* node" — following them answers PRD-05's
#: "impact of change" for nodes that are not consumed through an operation port
#: (the sample a run ``measured_sample``, a setup/reagent/calibration a result
#: ``linked_to`` / ``parameter_depends_on`` / ``uses_external_reference``). The
#: operation-graph relationships (``produced``/``input_to``) are excluded — that
#: derivation is followed natively via ``mfdb_operation_artifact`` — as are the
#: pure-membership relationships (``contains``/``included_in``/``grouped_in``/
#: ``project_contains``/``supersedes``), which are organisational, not provenance.
#: ``calibrated_by`` is listed forward (PRD-05 adds the vocabulary term and the
#: edge writes); filtering on a not-yet-emitted name is simply a no-op today.
USAGE_RELATIONSHIPS: frozenset[str] = frozenset(
    {
        "calibrated_by",
        "measured_sample",
        "parameter_depends_on",
        "linked_to",
        "uses_external_reference",
    }
)


@dataclass(frozen=True)
class LineageNode:
    """A node in a provenance graph (an artifact or an operation)."""

    node_type: str  # "artifact" | "operation"
    node_id: str
    label: str = ""  # artifact_kind or operation_type when known


@dataclass(frozen=True)
class LineageEdge:
    """A directed produced/consumed edge between an operation and an artifact."""

    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relationship: str  # "produced" | "input_to"


class Lineage:
    """Read-only lineage queries over ``mfdb_operation_artifact``."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "Lineage":
        return cls(conn)

    @classmethod
    def from_db(cls, db: Any) -> "Lineage":
        """Build from an ``MFDatabase`` (uses its live ``conn``)."""
        return cls(db.conn)

    # -- low-level neighbour steps over the operation graph -------------------

    def _column(self, sql: str, params: Iterable[Any]) -> list[str]:
        return [row[0] for row in self.conn.execute(sql, list(params)).fetchall()]

    def _producing_ops(self, artifact_id: str) -> list[str]:
        """Operations that output ``artifact_id``."""
        return self._column(
            "SELECT DISTINCT operation_id FROM mfdb_operation_artifact "
            "WHERE artifact_id = ? AND direction = 'output'",
            (artifact_id,),
        )

    def _consuming_ops(self, artifact_id: str) -> list[str]:
        """Operations that take ``artifact_id`` as input."""
        return self._column(
            "SELECT DISTINCT operation_id FROM mfdb_operation_artifact "
            "WHERE artifact_id = ? AND direction = 'input'",
            (artifact_id,),
        )

    def _op_inputs(self, operation_id: str) -> list[str]:
        return self._column(
            "SELECT DISTINCT artifact_id FROM mfdb_operation_artifact "
            "WHERE operation_id = ? AND direction = 'input'",
            (operation_id,),
        )

    def _op_outputs(self, operation_id: str) -> list[str]:
        return self._column(
            "SELECT DISTINCT artifact_id FROM mfdb_operation_artifact "
            "WHERE operation_id = ? AND direction = 'output'",
            (operation_id,),
        )

    def _step(self, artifact_id: str, *, upstream: bool) -> list[str]:
        """Adjacent artifacts one operation-hop away (parents or children)."""
        if upstream:
            ops = self._producing_ops(artifact_id)
            reach = self._op_inputs
        else:
            ops = self._consuming_ops(artifact_id)
            reach = self._op_outputs
        out: list[str] = []
        for op in ops:
            for nbr in reach(op):
                if nbr != artifact_id:
                    out.append(nbr)
        return out

    def _transitive(self, artifact_id: str, *, upstream: bool, max_depth: int) -> list[str]:
        """BFS closure of adjacent artifacts; derivation order, no duplicates."""
        seen: set[str] = {artifact_id}
        ordered: list[str] = []
        frontier = [artifact_id]
        depth = 0
        while frontier and depth < max_depth:
            nxt: list[str] = []
            for aid in frontier:
                for nbr in self._step(aid, upstream=upstream):
                    if nbr not in seen:
                        seen.add(nbr)
                        ordered.append(nbr)
                        nxt.append(nbr)
            frontier = nxt
            depth += 1
        return ordered

    # -- public API ----------------------------------------------------------

    def ancestors(self, artifact_id: str, *, max_depth: int = 100) -> list[str]:
        """Artifacts ``artifact_id`` was (transitively) derived from."""
        return self._transitive(artifact_id, upstream=True, max_depth=max_depth)

    def descendants(self, artifact_id: str, *, max_depth: int = 100) -> list[str]:
        """Artifacts (transitively) derived from ``artifact_id``."""
        return self._transitive(artifact_id, upstream=False, max_depth=max_depth)

    def lineage_to_root(self, artifact_id: str, *, max_depth: int = 100) -> list[str]:
        """The derivation chain: ``artifact_id`` first, then its ancestors."""
        return [artifact_id, *self.ancestors(artifact_id, max_depth=max_depth)]

    def what_used(self, node_id: str, *, max_depth: int = 100) -> list[str]:
        """Artifacts impacted if ``node_id`` changes (PRD-05's "impact of change").

        This is the full impact query and works for any node, not just artifacts
        consumed through an operation port:

        - the transitive **operation-graph descendants** of ``node_id`` (a raw or
          processed artifact consumed as an operation input), plus
        - the consumers reached over :data:`USAGE_RELATIONSHIPS` ``mfdb_edge`` links
          (a fit ``calibrated_by`` this calibration, a run that ``measured_sample``
          this sample, a result that ``linked_to`` this setup/reagent) and *their*
          descendants.

        Deduplicated; operation-graph descendants first. A plain artifact with no
        usage edges therefore still returns exactly its descendants.
        """
        return self.impact_of(node_id, max_depth=max_depth)

    def _edge_referrers(
        self, node_id: str, *, relationships: Iterable[str]
    ) -> list[tuple[str, str]]:
        """``(source_type, source_id)`` of non-operation edges *into* ``node_id``.

        The direct consumers of ``node_id``: rows whose ``target_node_id`` is
        ``node_id`` and whose ``relationship_type`` is one of ``relationships``
        (soft-deleted edges excluded).
        """
        rels = tuple(relationships)
        if not rels:
            return []
        placeholders = ",".join("?" for _ in rels)
        rows = self.conn.execute(
            "SELECT DISTINCT source_node_type, source_node_id FROM mfdb_edge "
            "WHERE target_node_id = ? AND deleted_at IS NULL "
            f"AND relationship_type IN ({placeholders})",
            (node_id, *rels),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def impact_of(
        self,
        node_id: str,
        *,
        relationships: Iterable[str] = USAGE_RELATIONSHIPS,
        max_depth: int = 100,
    ) -> list[str]:
        """Artifact ids impacted by a change to ``node_id`` (see :meth:`what_used`).

        Generalises ``descendants`` beyond the operation graph by also following
        ``mfdb_edge`` usage relationships to consumers (operations contribute their
        output artifacts; artifacts contribute themselves) and adding each consumer's
        descendants. Pass ``relationships`` to scope which links count as "usage".
        """
        impacted: list[str] = []
        seen: set[str] = set()

        def _add(aid: str) -> None:
            if aid not in seen:
                seen.add(aid)
                impacted.append(aid)
            for d in self.descendants(aid, max_depth=max_depth):
                if d not in seen:
                    seen.add(d)
                    impacted.append(d)

        # 1. operation-graph descendants (node consumed as an operation input)
        for d in self.descendants(node_id, max_depth=max_depth):
            if d not in seen:
                seen.add(d)
                impacted.append(d)

        # 2. edge referrers (calibrated_by / measured_sample / linked_to / …)
        _operation_types = {"operation", "processing_run", "analysis_run"}
        for src_type, src_id in self._edge_referrers(
            node_id, relationships=relationships
        ):
            if (src_type or "").lower() in _operation_types:
                for out in self._op_outputs(src_id):
                    _add(out)
            else:
                _add(src_id)

        return impacted

    def parents(self, artifact_id: str) -> list[str]:
        """Direct (one-hop) source artifacts."""
        return self._step(artifact_id, upstream=True)

    def children(self, artifact_id: str) -> list[str]:
        """Direct (one-hop) derived artifacts."""
        return self._step(artifact_id, upstream=False)

    # -- graph projection (for visualization) --------------------------------

    def _artifact_label(self, artifact_id: str) -> str:
        row = self.conn.execute(
            "SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return (row[0] if row else "") or ""

    def _operation_label(self, operation_id: str) -> str:
        row = self.conn.execute(
            "SELECT operation_type FROM mfdb_operation WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        return (row[0] if row else "") or ""

    def provenance_graph(
        self, artifact_id: str, *, depth: int = 100
    ) -> dict[str, list[dict[str, Any]]]:
        """Return ``{"nodes": [...], "edges": [...]}`` around ``artifact_id``.

        Walks both directions over the operation graph to ``depth`` operation-hops,
        emitting artifact and operation nodes plus ``produced``/``input_to`` edges —
        suitable for the admin lineage view / pipeline GUI.
        """
        nodes: dict[tuple[str, str], LineageNode] = {}
        edges: dict[tuple[str, str, str, str, str], LineageEdge] = {}

        def add_artifact(aid: str) -> None:
            key = ("artifact", aid)
            if key not in nodes:
                nodes[key] = LineageNode("artifact", aid, self._artifact_label(aid))

        def add_operation(oid: str) -> None:
            key = ("operation", oid)
            if key not in nodes:
                nodes[key] = LineageNode("operation", oid, self._operation_label(oid))

        def add_edge(e: LineageEdge) -> None:
            edges[(e.source_type, e.source_id, e.target_type, e.target_id, e.relationship)] = e

        add_artifact(artifact_id)
        seen: set[str] = set()
        frontier = [artifact_id]
        hops = 0
        while frontier and hops < depth:
            nxt: list[str] = []
            for aid in frontier:
                if aid in seen:
                    continue
                seen.add(aid)
                # producing operations: op --produced--> artifact
                for op in self._producing_ops(aid):
                    add_operation(op)
                    add_edge(LineageEdge("operation", op, "artifact", aid, "produced"))
                    for src in self._op_inputs(op):
                        add_artifact(src)
                        add_edge(LineageEdge("artifact", src, "operation", op, "input_to"))
                        nxt.append(src)
                # consuming operations: artifact --input_to--> op --produced--> child
                for op in self._consuming_ops(aid):
                    add_operation(op)
                    add_edge(LineageEdge("artifact", aid, "operation", op, "input_to"))
                    for out in self._op_outputs(op):
                        add_artifact(out)
                        add_edge(LineageEdge("operation", op, "artifact", out, "produced"))
                        nxt.append(out)
            frontier = nxt
            hops += 1

        return {
            "nodes": [
                {"node_type": n.node_type, "node_id": n.node_id, "label": n.label}
                for n in nodes.values()
            ],
            "edges": [
                {
                    "source_type": e.source_type,
                    "source_id": e.source_id,
                    "target_type": e.target_type,
                    "target_id": e.target_id,
                    "relationship": e.relationship,
                }
                for e in edges.values()
            ],
        }
