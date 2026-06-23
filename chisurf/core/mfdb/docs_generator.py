"""Generate an MFDB schema/API reference from the dictionary (PRD-26 Task 5).

The ``.dic`` is the single source of truth for the schema (PRD-19) and operation
parameter schemas (PRD-11). This module renders that truth as a Markdown reference
— tables/columns/types/foreign-keys, controlled vocabularies, and operation
parameter schemas — so the documentation can never drift from the dictionary: it is
regenerated, not hand-maintained.

Usage
-----
>>> from chisurf.core.mfdb.docs_generator import generate_schema_reference
>>> md = generate_schema_reference()                       # all flr_/mfdb_ tables
>>> md = generate_schema_reference(tables=db.dao.columns)  # restrict to a live schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary

#: Operation parameter schemas authored for PRD-11 (seeded into
#: ``mfdb_operation_parameter_def``); rendered here as the canonical reference.
_OP_DEFS_PATH = Path(__file__).parent / "data" / "operation_parameter_defs.json"

#: Namespaces the default (no explicit ``tables``) generation includes.
_MFDB_TABLE_PREFIXES = ("flr_", "mfdb_")


def _table_and_column(item) -> tuple[str, str]:
    """Apply the dictionary→schema mapping rule (PRD-19) for one item."""
    return (item.schema_table or item.category, item.schema_column or item.attribute)


def _wanted(table: str, tables: set[str] | None) -> bool:
    if tables is not None:
        return table in tables
    return table.startswith(_MFDB_TABLE_PREFIXES)


def collect_schema(
    dictionary: MmcifDictionary,
    tables: Iterable[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{table: [column_meta, ...]}`` from the dictionary.

    ``column_meta`` carries ``column``, ``type``, ``required``, ``foreign_key``,
    ``enumerations``, ``enum_details`` and ``description``. When ``tables`` is given
    the result is restricted to those tables (e.g. a live MFDatabase schema);
    otherwise every ``flr_*``/``mfdb_*`` table the dictionary declares is included.
    """
    want = set(tables) if tables is not None else None
    out: dict[str, list[dict[str, Any]]] = {}
    for category_name in dictionary.categories():
        category = dictionary.get_category(category_name)
        if category is None:
            continue
        for item in category.items.values():
            table, column = _table_and_column(item)
            if not _wanted(table, want):
                continue
            out.setdefault(table, []).append(
                {
                    "column": column,
                    "type": item.type_code or "",
                    "required": bool(item.mandatory),
                    "foreign_key": item.schema_foreign_key or "",
                    "enumerations": list(item.enumerations or []),
                    "enum_details": dict(item.enum_details or {}),
                    "description": (item.description or "").strip(),
                }
            )
    for cols in out.values():
        cols.sort(key=lambda c: c["column"])
    return out


def _load_operation_defs() -> dict[str, list[dict[str, Any]]]:
    try:
        data = json.loads(_OP_DEFS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def _render_tables(schema: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    lines = ["## Tables", ""]
    for table in sorted(schema):
        cols = schema[table]
        lines.append(f"### `{table}`")
        lines.append("")
        lines.append("| Column | Type | Required | Foreign key | Allowed values | Description |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c in cols:
            enums = ", ".join(c["enumerations"][:8])
            if len(c["enumerations"]) > 8:
                enums += ", …"
            lines.append(
                f"| `{c['column']}` | {_md_escape(c['type']) or '—'} | "
                f"{'yes' if c['required'] else ''} | "
                f"{_md_escape(c['foreign_key']) or ''} | {_md_escape(enums)} | "
                f"{_md_escape(c['description'])} |"
            )
        lines.append("")
    return lines


def _render_vocabularies(schema: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    lines = ["## Controlled vocabularies", ""]
    any_vocab = False
    for table in sorted(schema):
        for c in schema[table]:
            if not c["enumerations"]:
                continue
            any_vocab = True
            lines.append(f"- **`{table}.{c['column']}`**: " + ", ".join(c["enumerations"]))
            for value, detail in (c["enum_details"] or {}).items():
                if detail:
                    lines.append(f"  - `{value}` — {_md_escape(detail)}")
    if not any_vocab:
        lines.append("_None declared._")
    lines.append("")
    return lines


def _render_operations(op_defs: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    lines = ["## Operation parameter schemas (PRD-11)", ""]
    if not op_defs:
        lines.append("_None authored._")
        lines.append("")
        return lines
    for op_type in sorted(op_defs):
        params = op_defs[op_type]
        lines.append(f"### `{op_type}`")
        lines.append("")
        lines.append("| Parameter | Type | Required | Repeatable | Units | Bounds | Default | Description |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for p in params:
            lo, hi = p.get("lower_bound"), p.get("upper_bound")
            bounds = ""
            if lo is not None or hi is not None:
                bounds = f"{lo if lo is not None else '−∞'}..{hi if hi is not None else '∞'}"
            lines.append(
                f"| `{p.get('name', '')}` | {p.get('value_type', '')} | "
                f"{'yes' if p.get('required') else ''} | "
                f"{'yes' if p.get('repeatable') else ''} | "
                f"{_md_escape(str(p.get('units', '') or ''))} | {bounds} | "
                f"{_md_escape(str(p.get('default_value', '') or ''))} | "
                f"{_md_escape(p.get('description', ''))} |"
            )
        lines.append("")
    return lines


def generate_schema_reference(
    *,
    dictionary: MmcifDictionary | None = None,
    tables: Iterable[str] | None = None,
    operation_defs: Mapping[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Render the MFDB schema/API reference as Markdown from the dictionary."""
    dictionary = dictionary or MmcifDictionary.load_bundled()
    schema = collect_schema(dictionary, tables=tables)
    op_defs = operation_defs if operation_defs is not None else _load_operation_defs()

    lines = [
        "# MFDB schema reference",
        "",
        "_Generated from the flrCIF + `mfdb_flr_ext.dic` dictionary by "
        "`chisurf.core.mfdb.docs_generator` (PRD-26). Do not edit by hand — regenerate._",
        "",
        f"Tables: {len(schema)} · "
        f"Operation types: {len(op_defs)}",
        "",
    ]
    lines += _render_tables(schema)
    lines += _render_vocabularies(schema)
    lines += _render_operations(op_defs)
    return "\n".join(lines).rstrip() + "\n"


def write_schema_reference(
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write the generated reference to ``path`` and return it."""
    target = Path(path)
    target.write_text(generate_schema_reference(**kwargs), encoding="utf-8")
    return target
