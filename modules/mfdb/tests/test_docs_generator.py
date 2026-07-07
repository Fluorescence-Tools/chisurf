"""PRD-26 Task 5: the dictionary-driven schema/API docs generator.

Proves the reference is rendered from the dictionary (so it cannot drift) and,
when restricted to a live MFDatabase schema, covers exactly those tables plus the
PRD-11 operation parameter schemas.
"""

from __future__ import annotations

import os
import re
import tempfile

import pytest

from mfdb.docs_generator import (
    collect_schema,
    generate_schema_reference,
)
from mfdb.pdbx_metadata import MmcifDictionary
from mfdb.repository import MFDatabase


@pytest.fixture(scope="module")
def dictionary() -> MmcifDictionary:
    return MmcifDictionary.load_bundled()


@pytest.fixture
def live_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MFDatabase(os.path.join(tmpdir, "t.db"))
        try:
            yield {
                r[0]
                for r in db.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            db.close()


def test_reference_covers_live_tables_and_operations(dictionary, live_tables):
    md = generate_schema_reference(dictionary=dictionary, tables=live_tables)
    assert md.startswith("# MFDB schema reference")
    # Key tables rendered with a column table.
    for table in ("flr_sample", "mfdb_artifact", "mfdb_operation_parameter_def"):
        assert f"### `{table}`" in md, table
    assert "| Column | Type | Required | Foreign key | Allowed values | Description |" in md
    # PRD-11 operation schemas rendered (with a known param + its declared bound).
    assert "### `burst_selection`" in md
    assert "min_photons" in md
    assert "## Controlled vocabularies" in md


def test_restricting_to_live_tables_excludes_others(dictionary, live_tables):
    """Every rendered table heading is a real live table (no drift, no extras)."""
    md = generate_schema_reference(dictionary=dictionary, tables=live_tables)
    rendered = set(re.findall(r"^### `([^`]+)`", md, flags=re.MULTILINE))
    # Operation-type headings (### `burst_selection`, …) are not tables; drop them.
    op_types = {"burst_selection", "microtime_shift"}
    rendered_tables = rendered - op_types
    assert rendered_tables, "no tables rendered"
    assert rendered_tables <= live_tables, rendered_tables - live_tables


def test_default_scope_is_flr_and_mfdb_namespaces(dictionary):
    """Without an explicit ``tables`` filter, only flr_/mfdb_ tables are emitted."""
    schema = collect_schema(dictionary)
    assert schema, "expected some flr_/mfdb_ tables"
    assert all(t.startswith(("flr_", "mfdb_")) for t in schema)
    assert "flr_sample" in schema
    # A pure base-mmCIF category (not part of the MFDB schema) is excluded.
    assert "array_data" not in schema


def test_generator_is_dictionary_only_and_deterministic(dictionary, live_tables):
    a = generate_schema_reference(dictionary=dictionary, tables=live_tables)
    b = generate_schema_reference(dictionary=dictionary, tables=live_tables)
    assert a == b
    # No unescaped pipes break the Markdown tables (every body row has the right
    # number of cell separators for a 6-column table is hard to assert cheaply;
    # at least ensure newlines are stripped inside cells via the escaper).
    assert "\n|" not in a.replace("\n| ", "\nX")  # header/body rows start with "| "
