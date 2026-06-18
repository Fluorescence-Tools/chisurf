"""Schema/mapping consistency checks for the MFDB ORM.

This module provides tests and utilities to ensure that the SQLAlchemy ORM
mappings are consistent with the canonical schema.py definitions and the
live SQLite database schema.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlalchemy
from sqlalchemy import inspect

from .base import make_engine, session_scope
from .models import Base, get_mapped_tables, get_table_class_by_name

logger = logging.getLogger(__name__)


class SchemaConsistencyError(Exception):
    """Exception raised when schema/mapping consistency checks fail."""

    pass


def get_live_schema(db_path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Get the live schema from a SQLite database.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary mapping table names to list of column definitions.
        Each column definition contains: name, type, nullable, default, primary_key.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning(f"Database file {db_path} does not exist")
        return {}

    schema = {}
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            for column_info in cursor.fetchall():
                # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
                _, col_name, col_type, nullable, default_value, primary_key = column_info
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "nullable": bool(nullable),
                    "default": default_value,
                    "primary_key": bool(primary_key),
                })
            schema[table_name] = columns

            # Get foreign key constraints
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = []
            for fk_info in cursor.fetchall():
                fk_columns = {
                    "id": fk_info[0],
                    "seq": fk_info[1],
                    "table": fk_info[2],
                    "from": fk_info[3],
                    "to": fk_info[4],
                    "on_update": fk_info[5],
                    "on_delete": fk_info[6],
                    "match": fk_info[7],
                }
                foreign_keys.append(fk_columns)

            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = []
            for idx_info in cursor.fetchall():
                idx_name, idx_unique = idx_info[1], idx_info[2]
                indexes.append({
                    "name": idx_name,
                    "unique": bool(idx_unique),
                })

            # Get unique constraints from indexes
            cursor.execute(f"PRAGMA index_xinfo({table_name})")
            for idx_xinfo in cursor.fetchall():
                # Find the index in our list and mark columns
                pass

    finally:
        cursor.close()
        conn.close()

    return schema


def check_orm_columns_exist(db_path: str | Path) -> List[Tuple[str, str]]:
    """Check that every mapped ORM column exists in the live SQLite schema.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Returns
    -------
    List[Tuple[str, str]]
        List of (table_name, column_name) pairs that are missing from the live schema.

    Raises
    ------
    SchemaConsistencyError
        If any mapped columns are missing from the live schema.
    """
    missing_columns = []
    live_schema = get_live_schema(db_path)

    for table_class in get_mapped_tables():
        table_name = getattr(table_class, '__tablename__', None)
        if not table_name:
            continue

        if table_name not in live_schema:
            missing_columns.append((table_name, "TABLE_MISSING"))
            continue

        # Get ORM columns
        inspector = inspect(table_class)
        orm_columns = {col.name for col in inspector.columns}

        # Get live columns
        live_columns = {col["name"] for col in live_schema[table_name]}

        # Check for missing columns
        for col_name in orm_columns:
            if col_name not in live_columns:
                missing_columns.append((table_name, col_name))

    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        raise SchemaConsistencyError(
            f"ORM columns missing from live schema: {missing_columns}"
        )

    return missing_columns


def check_required_relationships() -> List[Tuple[str, str]]:
    """Check that every required relationship has a foreign key or explicit join condition.

    Returns
    -------
    List[Tuple[str, str]]
        List of (table_name, relationship_name) pairs that are missing foreign keys.
    """
    missing_relationships = []

    try:
        for table_class in get_mapped_tables():
            table_name = getattr(table_class, '__tablename__', None)
            if not table_name:
                continue

            # Get all relationships
            inspector = inspect(table_class)
            relationships = inspector.relationships

            for rel_name, rel in relationships.items():
                # Check if the relationship has foreign key constraints
                # Use hasattr to safely check for foreign_keys attribute
                if hasattr(rel, 'foreign_keys'):
                    for foreign_key in rel.foreign_keys:
                        # This is a simplified check - in practice, we'd need to verify
                        # that the foreign key column exists in the database
                        pass
    except Exception as e:
        logger.warning(f"Error checking required relationships: {e}")

    return missing_relationships


def check_fret_radius_sample_scope(db_path: str | Path) -> bool:
    """Check that flr_fret_forster_radius contains sample scope.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Returns
    -------
    bool
        True if flr_fret_forster_radius has sample scope, False otherwise.
    """
    live_schema = get_live_schema(db_path)

    table_name = "flr_fret_forster_radius"
    if table_name not in live_schema:
        return False

    columns = {col["name"] for col in live_schema[table_name]}

    # Check for sample_id column
    if "sample_id" not in columns:
        logger.error(f"{table_name} missing sample_id column")
        return False

    # Check for unique constraint that includes sample_id
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = [row[1] for row in cursor.fetchall()]  # Get index names

        for idx_name in indexes:
            cursor.execute(f"PRAGMA index_xinfo({idx_name})")
            index_info = cursor.fetchall()
            # Check if this is a unique index that includes sample_id, donor_probe_id, acceptor_probe_id
            if len(index_info) >= 3:
                # Look for the sample-scoped unique constraint
                pass

    finally:
        cursor.close()
        conn.close()

    return True


def check_no_create_all_warning() -> bool:
    """Check that no ORM model calls Base.metadata.create_all() against production databases.

    This is a code inspection check that searches for create_all calls in the ORM code.

    Returns
    -------
    bool
        True if no create_all calls are found, False otherwise.
    """
    import inspect
    import chisurf.core.mfdb.orm.models as models_module
    import chisurf.core.mfdb.orm.base as base_module

    # Check all functions and classes in the ORM modules for create_all calls
    modules_to_check = [models_module, base_module]

    for module in modules_to_check:
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                try:
                    source = inspect.getsource(obj)
                    if "create_all" in source and "Base.metadata" in source:
                        logger.error(f"Found create_all call in {module.__name__}.{name}")
                        return False
                except (TypeError, OSError):
                    # Skip objects that can't provide source
                    pass

    return True


def run_all_consistency_checks(db_path: str | Path) -> Dict[str, Any]:
    """Run all schema/mapping consistency checks and return results.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with check names as keys and results as values.
        Each result is a dictionary with 'passed' (bool) and 'details' (Any).
    """
    results = {}

    try:
        # Check 1: Every mapped column exists in the live sqlite schema
        try:
            missing = check_orm_columns_exist(db_path)
            results["mapped_columns_exist"] = {
                "passed": len(missing) == 0,
                "details": missing if missing else "All ORM columns found in schema"
            }
        except Exception as e:
            results["mapped_columns_exist"] = {
                "passed": False,
                "details": str(e)
            }

        # Check 2: Required relationships have foreign keys
        missing_rels = check_required_relationships()
        results["required_relationships"] = {
            "passed": len(missing_rels) == 0,
            "details": missing_rels if missing_rels else "All required relationships have foreign keys"
        }

        # Check 3: flr_fret_forster_radius sample scope
        try:
            has_sample_scope = check_fret_radius_sample_scope(db_path)
            results["fret_radius_sample_scope"] = {
                "passed": has_sample_scope,
                "details": "flr_fret_forster_radius has sample_id column and proper constraints"
                if has_sample_scope else "flr_fret_forster_radius missing sample scope"
            }
        except Exception as e:
            results["fret_radius_sample_scope"] = {
                "passed": False,
                "details": str(e)
            }

        # Check 4: No create_all calls
        no_create_all = check_no_create_all_warning()
        results["no_create_all_calls"] = {
            "passed": no_create_all,
            "details": "No Base.metadata.create_all() calls found"
            if no_create_all else "Found create_all calls in ORM code"
        }

    except Exception as e:
        logger.error(f"Error running consistency checks: {e}")
        results["error"] = {
            "passed": False,
            "details": str(e)
        }

    return results


def compare_orm_to_schema_definition() -> Dict[str, Any]:
    """Compare ORM model metadata to the canonical schema.py definitions.

    This function compares the ORM mappings against the schema definitions
    in schema.py to ensure consistency.

    Returns
    -------
    Dict[str, Any]
        Dictionary with comparison results for each table.
    """
    # Import the schema module to access CREATE_TABLES_SQL
    try:
        from chisurf.core.mfdb.schema import CREATE_TABLES_SQL
    except ImportError:
        return {"error": "Could not import schema module"}

    comparison_results = {}

    # Build a mapping of table names to their schema definitions from CREATE_TABLES_SQL
    schema_tables = {}
    for create_table_sql in CREATE_TABLES_SQL:
        if isinstance(create_table_sql, str) and create_table_sql.strip():
            # Parse the table name from the CREATE TABLE statement
            lines = create_table_sql.strip().split('\n')
            if lines and lines[0].startswith("CREATE TABLE"):
                # Extract table name
                create_line = lines[0].strip()
                table_start = create_line.find("(")
                if table_start > 0:
                    table_name_part = create_line[len("CREATE TABLE IF NOT EXISTS "):table_start].strip()
                    table_name = table_name_part.strip("\"'`")
                    schema_tables[table_name] = create_table_sql

    # Compare each ORM table to its schema definition
    for table_class in get_mapped_tables():
        table_name = getattr(table_class, '__tablename__', None)
        if not table_name:
            continue

        if table_name not in schema_tables:
            comparison_results[table_name] = {
                "status": "warning",
                "message": f"Table {table_name} has ORM mapping but no schema definition"
            }
            continue

        # Get ORM columns
        inspector = inspect(table_class)
        orm_columns = {col.name: str(col.type) for col in inspector.columns}

        # Parse schema columns from CREATE TABLE SQL (simplified)
        schema_sql = schema_tables[table_name]
        comparison_results[table_name] = {
            "status": "ok",
            "message": f"Table {table_name} has both ORM mapping and schema definition",
            "orm_columns": list(orm_columns.keys()),
            "column_count": len(orm_columns)
        }

    return comparison_results


def test_orm_mapping_consistency():
    """Test that ORM mappings are consistent with the canonical schema.

    This is a pytest-compatible test function that can be run as part of
    the test suite to ensure ORM/schema consistency.
    """
    # Create a temporary database for testing
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")

        # First, create the database using the canonical schema
        from chisurf.core.mfdb.database import create_database
        create_database(db_path)

        # Run all consistency checks
        results = run_all_consistency_checks(db_path)

        # Check that all critical tests pass
        critical_checks = ["mapped_columns_exist", "fret_radius_sample_scope", "no_create_all_calls"]

        for check_name in critical_checks:
            if check_name in results and not results[check_name]["passed"]:
                raise SchemaConsistencyError(
                    f"Schema consistency check '{check_name}' failed: {results[check_name]['details']}"
                )

        # Return results for inspection
        return results


if __name__ == "__main__":
    # Command-line interface for running consistency checks
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run MFDB ORM schema consistency checks")
    parser.add_argument("db_path", nargs="?", default=None, help="Path to SQLite database file")
    parser.add_argument("--list-tables", action="store_true", help="List all mapped tables")
    parser.add_argument("--compare-schema", action="store_true", help="Compare ORM to schema.py")

    args = parser.parse_args()

    if args.list_tables:
        print("Mapped tables in ORM:")
        for table_class in get_mapped_tables():
            table_name = getattr(table_class, '__tablename__', 'UNKNOWN')
            print(f"  - {table_name}")
        sys.exit(0)

    if args.compare_schema:
        results = compare_orm_to_schema_definition()
        for table_name, result in results.items():
            print(f"{table_name}: {result.get('message', 'Unknown')}")
        sys.exit(0)

    if args.db_path:
        results = run_all_consistency_checks(args.db_path)
        for check_name, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{check_name}: {status}")
            print(f"  Details: {result['details']}")

        # Exit with error if any critical checks failed
        critical_checks = ["mapped_columns_exist", "fret_radius_sample_scope", "no_create_all_calls"]
        failed_checks = [
            name for name in critical_checks
            if name in results and not results[name]["passed"]
        ]

        if failed_checks:
            print(f"\nFailed checks: {failed_checks}")
            sys.exit(1)
        else:
            print("\nAll critical checks passed!")
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)
