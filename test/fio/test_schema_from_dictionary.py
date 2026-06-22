"""Tests for schema_from_dictionary.py — DDL generation from the .dic.

Verifies that:
1. Generated CREATE TABLE contains expected columns, types, PK, FK.
2. A DB built via the generator has the same columns the .dic declares
   (round-trip with ``introspect_sqlite_schema``).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

from chisurf.core.mfdb.dictionary_schema_map import introspect_sqlite_schema
from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
from chisurf.core.mfdb.schema_from_dictionary import (
    TYPE_CODE_SQL_MAP,
    generate_create_table_for_category,
    generate_index_for_table,
)


def _dic() -> MmcifDictionary:
    return MmcifDictionary.load_bundled()


def _type_code_to_sql(tc: str) -> str:
    return TYPE_CODE_SQL_MAP.get(tc.lower(), "TEXT")


def _expected_sql_type(item) -> str:
    return _type_code_to_sql(item.type_code or "line")


def test_detector_channel_ddl_has_expected_columns() -> None:
    """Generated DDL for mfdb_setup_detector_channel contains all .dic columns."""
    dic = _dic()
    ddl = generate_create_table_for_category(dic, "mfdb_setup_detector_channel")
    assert "CREATE TABLE IF NOT EXISTS mfdb_setup_detector_channel" in ddl

    cat = dic.get_category("mfdb_setup_detector_channel")
    for item in cat.items.values():
        col = item.schema_column or item.attribute
        assert col in ddl, f"Column {col!r} missing from DDL"
        sql_type = _expected_sql_type(item)
        assert sql_type in ddl.split(col)[1].split()[0], (
            f"Column {col} has wrong type (expected {sql_type})"
        )

    # PK
    assert "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT" in ddl
    # FK (setup_id)
    assert "REFERENCES mfdb_setup(setup_id) ON DELETE CASCADE" in ddl
    # Audit columns
    assert "created_at" in ddl
    assert "updated_at" in ddl
    assert "deleted_at" in ddl


def test_pie_window_ddl_has_expected_columns() -> None:
    """Generated DDL for mfdb_setup_pie_window contains all .dic columns."""
    dic = _dic()
    ddl = generate_create_table_for_category(dic, "mfdb_setup_pie_window")
    assert "CREATE TABLE IF NOT EXISTS mfdb_setup_pie_window" in ddl

    cat = dic.get_category("mfdb_setup_pie_window")
    for item in cat.items.values():
        col = item.schema_column or item.attribute
        assert col in ddl, f"Column {col!r} missing from DDL"

    # PK
    assert "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT" in ddl
    # FK
    assert "REFERENCES mfdb_setup(setup_id) ON DELETE CASCADE" in ddl
    # NOT NULL on mandatory items
    assert "end INTEGER NOT NULL" in ddl
    assert "start INTEGER NOT NULL" in ddl


def test_round_trip_ddl_builds_db_with_dic_columns() -> None:
    """A DB built via the generated DDL has exactly the columns the .dic declares."""
    dic = _dic()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_roundtrip.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create mfdb_setup first (parent table for FK)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mfdb_setup (
            setup_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    # Create the child tables from generated DDL
    for cat_name in ["mfdb_setup_detector_channel", "mfdb_setup_pie_window"]:
        ddl = generate_create_table_for_category(dic, cat_name)
        cursor.execute(ddl)

    conn.commit()
    conn.close()

    # Introspect the live schema
    schema = introspect_sqlite_schema(db_path)

    for cat_name in ["mfdb_setup_detector_channel", "mfdb_setup_pie_window"]:
        cat = dic.get_category(cat_name)
        table_name = cat_name
        assert table_name in schema, f"Table {table_name} not found in DB"

        live_cols = set(schema[table_name].keys())
        for item in cat.items.values():
            col = item.schema_column or item.attribute
            assert col in live_cols, (
                f"Column {table_name}.{col} declared in .dic but missing in live DB"
            )
            actual_type = schema[table_name][col]["type"]
            expected_type = _expected_sql_type(item)
            assert actual_type == expected_type, (
                f"Column {table_name}.{col} type mismatch: "
                f"expected {expected_type}, got {actual_type}"
            )

        # Audit columns should be present
        for audit in ["created_at", "updated_at", "deleted_at"]:
            assert audit in live_cols, (
                f"Audit column {table_name}.{audit} missing from live DB"
            )


def test_generate_index() -> None:
    """generate_index_for_table produces valid DDL."""
    ddl = generate_index_for_table("mfdb_setup_detector_channel", "setup_id")
    assert ddl == (
        "CREATE INDEX IF NOT EXISTS idx_mfdb_setup_detector_channel_setup_id "
        "ON mfdb_setup_detector_channel (setup_id)"
    )
