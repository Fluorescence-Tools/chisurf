"""Generate SQL DDL from the mmCIF/flrCIF dictionary.

The dictionary is the single authored artifact for the reading/processing
setup tables (mfdb_setup_detector_channel, mfdb_setup_pie_window) and the
TTTR-reading columns on mfdb_setup.  Every column, type, PK, FK, and
default is derived from the ``_chisurf_schema`` bridge attributes.

Usage::

    from chisurf.core.mfdb.schema_from_dictionary import (
        generate_create_table_for_category,
        generate_alter_add_columns_for_category,
    )

    dic = MmcifDictionary.load_bundled()
    ddl = generate_create_table_for_category(dic, "mfdb_setup_detector_channel")
"""

from __future__ import annotations

from typing import Any

from chisurf.core.mfdb.pdbx_metadata import DictItem, MmcifDictionary

# ---------------------------------------------------------------------------
# SQL type map — counterpart of TYPE_CODE_WIDGET_MAP in entity_schema.py
# ---------------------------------------------------------------------------
TYPE_CODE_SQL_MAP: dict[str, str] = {
    "int": "INTEGER",
    "uint": "INTEGER",
    "integer": "INTEGER",
    "float": "REAL",
    "double": "REAL",
    "num": "REAL",
    "text": "TEXT",
    "line": "TEXT",
    "code": "TEXT",
    "char": "TEXT",
    "ucode": "TEXT",
    "date": "TEXT",
    "datetime": "TEXT",
    "boolean": "INTEGER",
    "yes_no": "INTEGER",
    "enum": "TEXT",
    "yyyy-mm-dd": "TEXT",
}

# Audit columns appended to every generated table
_AUDIT_COLUMNS_SQL = [
    "created_at TEXT DEFAULT CURRENT_TIMESTAMP",
    "updated_at TEXT DEFAULT CURRENT_TIMESTAMP",
    "deleted_at TEXT",
]


def _sql_type_for_item(item: DictItem) -> str:
    """Return the SQL column type for a dictionary item."""
    tc = (item.type_code or "").lower()
    return TYPE_CODE_SQL_MAP.get(tc, "TEXT")


def _resolve_table_name(category_name: str, dic: MmcifDictionary) -> str:
    """Resolve the SQL table name for a dictionary category.

    Uses the first item's ``_chisurf_schema.table_name``, or falls back
    to the category name.
    """
    cat = dic.get_category(category_name)
    if cat is not None:
        for item in cat.items.values():
            if item.schema_table:
                return item.schema_table
    return category_name


def _sorted_items(cat: Any) -> list[DictItem]:
    """Return category items in a stable order: PK first, then alphabetically."""
    pk_attr = _pk_attribute(cat)
    items = sorted(cat.items.values(), key=lambda i: (
        0 if i.attribute == pk_attr else 1,
        i.attribute or "",
    ))
    return items


def _pk_attribute(cat: Any) -> str:
    """Extract the PK column name from a category's key_item."""
    key = (cat.key_item or "").strip()
    if "." in key:
        return key.split(".")[-1]
    return key.replace("_", "").lower() or "id"


def generate_alter_add_columns_for_category(
    dic: MmcifDictionary,
    table_name: str,
    category_name: str,
) -> list[str]:
    """Generate ``ALTER TABLE ADD COLUMN`` statements for a category's items.

    Only items that map to the given *table_name* are included.
    """
    cat = dic.get_category(category_name)
    if cat is None:
        return []

    stmts: list[str] = []
    for item in _sorted_items(cat):
        t = item.schema_table or category_name
        if t != table_name:
            continue
        col = item.schema_column or item.attribute
        sql_type = _sql_type_for_item(item)
        nullable = " NOT NULL" if item.mandatory else ""
        default = ""
        if item.default_value:
            default = f" DEFAULT {_quote_default(item.default_value, sql_type)}"
        stmts.append(
            f"ALTER TABLE {table_name} ADD COLUMN {col} {sql_type}{nullable}{default}"
        )
    return stmts


def _quote_default(value: str, sql_type: str) -> str:
    """Quote a default value appropriately for the SQL type."""
    value = value.strip()
    if sql_type in ("INTEGER", "REAL"):
        return value
    return f"'{value.replace(chr(39), chr(39)*2)}'"


def _column_def(item: DictItem, category_name: str) -> str:
    """Build the ``column TYPE [NOT NULL] [DEFAULT ...]`` fragment."""
    col = item.schema_column or item.attribute
    sql_type = _sql_type_for_item(item)
    parts = [col, sql_type]

    if item.mandatory:
        parts.append("NOT NULL")

    if item.default_value:
        parts.append(f"DEFAULT {_quote_default(item.default_value, sql_type)}")

    return " ".join(parts)


def _fk_clause(item: DictItem) -> str:
    """Return a ``REFERENCES ...`` clause from ``schema_foreign_key``."""
    fk = (item.schema_foreign_key or "").strip()
    if fk:
        return f" REFERENCES {fk}"
    return ""


def generate_create_table_for_category(
    dic: MmcifDictionary,
    category_name: str,
) -> str:
    """Generate a ``CREATE TABLE`` statement for one dictionary category.

    Parameters
    ----------
    dic : MmcifDictionary
        Loaded dictionary.
    category_name : str
        Category name (e.g. ``"mfdb_setup_detector_channel"``).

    Returns
    -------
    str
        The ``CREATE TABLE`` DDL statement.
    """
    cat = dic.get_category(category_name)
    if cat is None:
        return f"-- Category {category_name!r} not found in dictionary"

    table_name = _resolve_table_name(category_name, dic)
    pk_attr = _pk_attribute(cat)

    lines = [f"CREATE TABLE IF NOT EXISTS {table_name} ("]
    col_lines: list[str] = []

    for item in _sorted_items(cat):
        col = item.schema_column or item.attribute
        cd = _column_def(item, category_name)
        fk = _fk_clause(item)

        if col == pk_attr:
            if pk_attr in ("id", "setup_id"):
                # setup_id is a TEXT PK (natural key); id is INTEGER AUTOINCREMENT
                if item.type_code and item.type_code.lower() in ("int", "integer"):
                    col_lines.append(f"    {cd} PRIMARY KEY AUTOINCREMENT{fk}")
                else:
                    col_lines.append(f"    {cd} PRIMARY KEY{fk}")
            else:
                col_lines.append(f"    {cd} PRIMARY KEY{fk}")
        else:
            col_lines.append(f"    {cd}{fk}")

    # Audit columns are appended by convention (not in the dictionary)
    seen = {item.schema_column or item.attribute for item in cat.items.values()}
    for audit_sql in _AUDIT_COLUMNS_SQL:
        audit_name = audit_sql.split()[0]
        if audit_name not in seen:
            col_lines.append(f"    {audit_sql}")

    lines.append(",\n".join(col_lines))
    lines.append(")")
    return "".join(lines)


def generate_index_for_table(
    table_name: str,
    column_name: str,
    index_name: str | None = None,
) -> str:
    """Generate a ``CREATE INDEX`` statement."""
    ix = index_name or f"idx_{table_name}_{column_name}"
    return f"CREATE INDEX IF NOT EXISTS {ix} ON {table_name} ({column_name})"
