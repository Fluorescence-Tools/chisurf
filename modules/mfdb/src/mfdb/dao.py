"""Dictionary-/schema-driven data access (PRD-26 Task 1).

The ``.dic`` already dictates the schema (PRD-19). This module is the generated
**DAO core** that the hand-written repository SQL can migrate onto: a generic,
fully-parameterised CRUD layer whose table/column identifiers are **whitelisted
against the live schema** (so identifiers can never carry caller input) and whose
values are always bound parameters (no f-string interpolation).

It is intentionally generic — one ``DictionaryDao`` works for every
dictionary-declared table — and consistent: soft-delete (``deleted_at``) and an
``updated_at`` touch are applied automatically when the table declares those
columns. Bespoke queries (lineage, browse, joins) stay hand-written; this layer
covers the CRUD-shaped majority.

Example
-------
>>> dao = DictionaryDao.from_connection(conn)
>>> dao.insert("flr_sample", {"sample_id": "s1", "description": "DNA"})
>>> dao.get("flr_sample", "s1")
{'sample_id': 's1', 'description': 'DNA', ...}
>>> dao.update("flr_sample", "s1", {"description": "DNA duplex"})
>>> dao.soft_delete("flr_sample", "s1")
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Mapping

from mfdb.dictionary_schema_map import (
    DictionarySchemaMap,
    introspect_sqlite_schema,
    quote_identifier,
)

#: Conventional columns the DAO applies automatically when a table declares them.
SOFT_DELETE_COLUMN = "deleted_at"
UPDATED_AT_COLUMN = "updated_at"


class DaoError(Exception):
    """Base error for dictionary-driven data access."""


class UnknownTableError(DaoError):
    """Raised when a table is not part of the live/declared schema."""


class UnknownColumnError(DaoError):
    """Raised when a column is not declared for the target table."""


class DictionaryDao:
    """Generic, parameterised CRUD over dictionary-declared MFDB tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection. ``row_factory`` is set to ``sqlite3.Row`` if unset so
        rows are returned as dicts.
    schema : Mapping[str, Mapping[str, Mapping[str, Any]]]
        ``{table: {column: column_meta}}`` as produced by
        :func:`introspect_sqlite_schema`. The keys define the identifier
        whitelist; values carry ``primary_key``/``notnull``/``type`` metadata.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        schema: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        self.conn = conn
        if getattr(conn, "row_factory", None) is None:
            conn.row_factory = sqlite3.Row
        self._schema = schema

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "DictionaryDao":
        """Build a DAO by introspecting the live schema of ``conn``."""
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        schema: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            name = row[0]
            columns: dict[str, dict[str, Any]] = {}
            for cid, col, col_type, notnull, default, pk in conn.execute(
                f"PRAGMA table_info({quote_identifier(name)})"
            ):
                columns[col] = {
                    "cid": cid,
                    "type": col_type,
                    "notnull": bool(notnull),
                    "default": default,
                    "primary_key": bool(pk),
                }
            schema[name] = columns
        return cls(conn, schema)

    @classmethod
    def from_dictionary_map(
        cls, conn: sqlite3.Connection, schema_map: DictionarySchemaMap
    ) -> "DictionaryDao":
        """Build a DAO from a :class:`DictionarySchemaMap`'s live schema.

        Falls back to introspecting ``conn`` when the map carries no live schema.
        """
        if schema_map.schema:
            return cls(conn, schema_map.schema)
        return cls.from_connection(conn)

    @classmethod
    def from_db_path(cls, conn: sqlite3.Connection, db_path: str) -> "DictionaryDao":
        """Build a DAO whose schema is introspected from ``db_path``."""
        return cls(conn, introspect_sqlite_schema(db_path))

    # -- schema helpers --------------------------------------------------

    def has_table(self, table: str) -> bool:
        """Return ``True`` if ``table`` is part of the declared schema."""
        return table in self._schema

    def columns(self, table: str) -> set[str]:
        """Return the declared column names for ``table``."""
        self._require_table(table)
        return set(self._schema[table])

    def primary_key(self, table: str) -> str:
        """Return the primary-key column for ``table``.

        Falls back to ``<table>_id`` / ``id`` when no PRAGMA primary key is set.
        """
        self._require_table(table)
        cols = self._schema[table]
        for name, meta in cols.items():
            if meta.get("primary_key"):
                return name
        for candidate in (f"{table}_id", "id", "uuid"):
            if candidate in cols:
                return candidate
        raise DaoError(f"No primary key found for table {table!r}.")

    def _require_table(self, table: str) -> None:
        if table not in self._schema:
            raise UnknownTableError(f"Unknown table: {table!r}.")

    def _require_columns(self, table: str, names: Iterable[str]) -> None:
        declared = self._schema[table]
        unknown = [n for n in names if n not in declared]
        if unknown:
            raise UnknownColumnError(
                f"Unknown column(s) for {table!r}: {sorted(unknown)}."
            )

    def _has(self, table: str, column: str) -> bool:
        return column in self._schema.get(table, {})

    # -- CRUD ------------------------------------------------------------

    def insert(self, table: str, values: Mapping[str, Any]) -> str | int | None:
        """Insert one row; return its primary-key value (or last rowid).

        All identifiers are whitelisted against the schema and all values are
        bound parameters.
        """
        self._require_table(table)
        if not values:
            raise DaoError(f"insert into {table!r} requires at least one value.")
        self._require_columns(table, values.keys())

        cols = list(values.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_sql = ", ".join(quote_identifier(c) for c in cols)
        sql = (
            f"INSERT INTO {quote_identifier(table)} ({col_sql}) VALUES ({placeholders})"
        )
        cur = self.conn.execute(sql, [values[c] for c in cols])
        pk = self.primary_key(table)
        if pk in values:
            return values[pk]
        return cur.lastrowid

    def get(
        self,
        table: str,
        pk_value: Any,
        *,
        pk_column: str | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Return one row by primary key as a dict, or ``None``."""
        self._require_table(table)
        pk = pk_column or self.primary_key(table)
        self._require_columns(table, [pk])
        sql = f"SELECT * FROM {quote_identifier(table)} WHERE {quote_identifier(pk)} = ?"
        params: list[Any] = [pk_value]
        if not include_deleted and self._has(table, SOFT_DELETE_COLUMN):
            sql += f" AND {quote_identifier(SOFT_DELETE_COLUMN)} IS NULL"
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def list(
        self,
        table: str,
        *,
        filters: Mapping[str, Any] | None = None,
        include_deleted: bool = False,
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return rows matching equality ``filters`` (all parameterised)."""
        self._require_table(table)
        clauses: list[str] = []
        params: list[Any] = []
        if filters:
            self._require_columns(table, filters.keys())
            for col, val in filters.items():
                if val is None:
                    clauses.append(f"{quote_identifier(col)} IS NULL")
                else:
                    clauses.append(f"{quote_identifier(col)} = ?")
                    params.append(val)
        if not include_deleted and self._has(table, SOFT_DELETE_COLUMN):
            clauses.append(f"{quote_identifier(SOFT_DELETE_COLUMN)} IS NULL")

        sql = f"SELECT * FROM {quote_identifier(table)}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order_by is not None:
            self._require_columns(table, [order_by])
            sql += f" ORDER BY {quote_identifier(order_by)}"
            sql += " DESC" if descending else " ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
            if offset is not None:
                sql += " OFFSET ?"
                params.append(int(offset))
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def update(
        self,
        table: str,
        pk_value: Any,
        values: Mapping[str, Any],
        *,
        pk_column: str | None = None,
        touch: bool = True,
    ) -> int:
        """Update one row by primary key; return the affected row count.

        Sets ``updated_at`` to ``CURRENT_TIMESTAMP`` when the table declares it
        and ``touch`` is true. The primary-key column cannot be updated.
        """
        self._require_table(table)
        if not values:
            return 0
        self._require_columns(table, values.keys())
        pk = pk_column or self.primary_key(table)
        if pk in values:
            raise DaoError(f"Cannot update primary key {pk!r} of {table!r}.")

        set_cols = list(values.keys())
        assignments = [f"{quote_identifier(c)} = ?" for c in set_cols]
        params: list[Any] = [values[c] for c in set_cols]
        if touch and self._has(table, UPDATED_AT_COLUMN):
            assignments.append(f"{quote_identifier(UPDATED_AT_COLUMN)} = CURRENT_TIMESTAMP")
        sql = (
            f"UPDATE {quote_identifier(table)} SET {', '.join(assignments)} "
            f"WHERE {quote_identifier(pk)} = ?"
        )
        params.append(pk_value)
        if self._has(table, SOFT_DELETE_COLUMN):
            sql += f" AND {quote_identifier(SOFT_DELETE_COLUMN)} IS NULL"
        return int(self.conn.execute(sql, params).rowcount)

    def soft_delete(
        self,
        table: str,
        pk_value: Any,
        *,
        pk_column: str | None = None,
        deleted_at: str | None = None,
    ) -> int:
        """Soft-delete one row (set ``deleted_at``); return affected row count.

        ``deleted_at`` defaults to SQLite ``CURRENT_TIMESTAMP``; pass an explicit
        value (e.g. an application ``_utc_now()`` ISO string) to control the stored
        marker. Falls back to a hard ``DELETE`` for tables without a ``deleted_at``
        column.
        """
        self._require_table(table)
        pk = pk_column or self.primary_key(table)
        if not self._has(table, SOFT_DELETE_COLUMN):
            sql = f"DELETE FROM {quote_identifier(table)} WHERE {quote_identifier(pk)} = ?"
            return int(self.conn.execute(sql, [pk_value]).rowcount)
        params: list[Any] = []
        if deleted_at is None:
            set_clause = f"{quote_identifier(SOFT_DELETE_COLUMN)} = CURRENT_TIMESTAMP"
        else:
            set_clause = f"{quote_identifier(SOFT_DELETE_COLUMN)} = ?"
            params.append(deleted_at)
        sql = (
            f"UPDATE {quote_identifier(table)} SET {set_clause} "
            f"WHERE {quote_identifier(pk)} = ? "
            f"AND {quote_identifier(SOFT_DELETE_COLUMN)} IS NULL"
        )
        params.append(pk_value)
        return int(self.conn.execute(sql, params).rowcount)
