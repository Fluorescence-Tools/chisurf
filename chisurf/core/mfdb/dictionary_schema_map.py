"""Programmatic dictionary-to-schema mapping registry.

The mmCIF/flrCIF dictionary files are the authority for item names. ChiSurf
adds local dictionary entries for fields it stores but upstream flrCIF does not
define, and local schema binding metadata for any current database column whose
name differs from the dictionary item attribute.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chisurf.core.mfdb.pdbx_metadata import DictItem, MmcifDictionary


@dataclass(frozen=True)
class MappedColumn:
    """Mapping from one dictionary item to one MFDB table column."""

    dictionary_name: str
    table_name: str
    column_name: str
    category: str
    attribute: str
    type_code: str = ""
    description: str = ""
    source: str = "direct"


@dataclass(frozen=True)
class UnmappedItem:
    """Dictionary item that cannot be mapped to the current live schema."""

    dictionary_name: str
    category: str
    attribute: str
    candidate_table: str
    candidate_column: str
    reason: str


class DictionarySchemaMap:
    """Map dictionary items to live MFDB columns.

    Mapping rules are intentionally mechanical:

    1. If the dictionary item declares ``_chisurf_schema.table_name`` or
       ``_chisurf_schema.column_name``, use those values.
    2. Otherwise use the dictionary category as table name and item attribute
       as column name.
    3. When a live schema is provided, keep only mappings that point to real
       live columns and record the rest as unmapped.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        dictionary: MmcifDictionary | None = None,
    ) -> None:
        """Create a dictionary/schema registry."""
        self.dictionary = dictionary or MmcifDictionary.load_bundled()
        self.schema: dict[str, dict[str, dict[str, Any]]] = {}
        self._mapped_columns: dict[str, MappedColumn] = {}
        self._unmapped_items: dict[str, UnmappedItem] = {}

        if db_path is not None:
            self.schema = introspect_sqlite_schema(db_path)

        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build mappings for flrCIF and explicitly bound ChiSurf dictionary items."""
        for category_name in self._mapped_categories():
            category = self.dictionary.get_category(category_name)
            if category is None:
                continue

            for item in category.items.values():
                table_name, column_name, source = self._candidate_for_item(item)
                if self.schema and not self._schema_has_column(table_name, column_name):
                    reason = (
                        "table missing"
                        if table_name not in self.schema
                        else "column missing"
                    )
                    self._unmapped_items[item.name] = UnmappedItem(
                        dictionary_name=item.name,
                        category=item.category,
                        attribute=item.attribute,
                        candidate_table=table_name,
                        candidate_column=column_name,
                        reason=reason,
                    )
                    continue

                self._mapped_columns[item.name] = MappedColumn(
                    dictionary_name=item.name,
                    table_name=table_name,
                    column_name=column_name,
                    category=item.category,
                    attribute=item.attribute,
                    type_code=item.type_code,
                    description=item.description,
                    source=source,
                )

    def _mapped_categories(self) -> list[str]:
        """Return categories eligible for dictionary-to-schema mapping."""
        categories = []
        for category_name in self.dictionary.categories():
            category = self.dictionary.get_category(category_name)
            if category is None:
                continue
            if category_name.startswith("flr_") or any(
                item.schema_table or item.schema_column
                for item in category.items.values()
            ):
                categories.append(category_name)
        return categories

    @staticmethod
    def _candidate_for_item(item: DictItem) -> tuple[str, str, str]:
        """Return table/column candidate declared by the dictionary item."""
        table_name = item.schema_table or item.category
        column_name = item.schema_column or item.attribute
        source = "dictionary" if item.schema_table or item.schema_column else "direct"
        return table_name, column_name, source

    def _schema_has_column(self, table_name: str, column_name: str) -> bool:
        """Return whether the live schema has table.column."""
        return table_name in self.schema and column_name in self.schema[table_name]

    def map_dictionary_item(self, full_name: str) -> MappedColumn | None:
        """Return the mapped live column for a dictionary item."""
        normalized = normalize_dictionary_name(full_name)
        return self._mapped_columns.get(normalized)

    def validate_mapping(self, full_name: str) -> tuple[bool, str]:
        """Validate that a dictionary item maps to a live database column."""
        normalized = normalize_dictionary_name(full_name)
        mapped = self._mapped_columns.get(normalized)
        if mapped is not None:
            if not self.schema:
                return True, ""
            if self._schema_has_column(mapped.table_name, mapped.column_name):
                return True, ""
            return (
                False,
                f"{normalized} maps to missing column "
                f"{mapped.table_name}.{mapped.column_name}",
            )

        unmapped = self._unmapped_items.get(normalized)
        if unmapped is not None:
            return (
                False,
                f"{normalized} is unmapped: {unmapped.reason} "
                f"({unmapped.candidate_table}.{unmapped.candidate_column})",
            )
        return False, f"Unknown dictionary item: {normalized}"

    def get_table_columns(self, table_name: str) -> list[str]:
        """Return live schema columns for one table."""
        return sorted(self.schema.get(table_name, {}))

    def get_flr_tables(self) -> list[str]:
        """Return live schema tables with the flr_ prefix."""
        return sorted(table for table in self.schema if table.startswith("flr_"))

    def get_mapped_flr_items(self) -> list[MappedColumn]:
        """Return all mapped flrCIF items."""
        return sorted(
            self._mapped_columns.values(),
            key=lambda item: item.dictionary_name,
        )

    def get_unmapped_flr_items(self) -> list[UnmappedItem]:
        """Return flrCIF items that do not map to the current live schema."""
        return sorted(
            self._unmapped_items.values(),
            key=lambda item: item.dictionary_name,
        )

    def get_schema_columns_for_category(
        self, category: str
    ) -> list[tuple[str, str, str]]:
        """Return mapped dictionary/table/column triples for one category."""
        return [
            (mapped.dictionary_name, mapped.table_name, mapped.column_name)
            for mapped in self.get_mapped_flr_items()
            if mapped.category == category
        ]

    def categories_without_live_table(self) -> list[str]:
        """Return flrCIF categories absent from the live MFDB schema."""
        if not self.schema:
            return []
        return sorted(
            category
            for category in self.dictionary.flr_categories()
            if category not in self.schema
            and not any(
                item.schema_table in self.schema
                for item in self.dictionary.get_category(category).items.values()
            )
        )


def normalize_dictionary_name(full_name: str) -> str:
    """Normalize ``category.attribute`` to ``_category.attribute`` form."""
    return full_name if full_name.startswith("_") else f"_{full_name}"


def introspect_sqlite_schema(db_path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return table/column metadata from a live SQLite database."""
    schema: dict[str, dict[str, dict[str, Any]]] = {}
    with sqlite3.connect(str(db_path)) as conn:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (table_name,) in table_rows:
            quoted = quote_identifier(table_name)
            columns: dict[str, dict[str, Any]] = {}
            for cid, name, col_type, notnull, default, primary_key in conn.execute(
                f"PRAGMA table_info({quoted})"
            ):
                columns[name] = {
                    "cid": cid,
                    "type": col_type,
                    "notnull": bool(notnull),
                    "default": default,
                    "primary_key": bool(primary_key),
                }
            schema[table_name] = columns
    return schema


def quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def build_dictionary_schema_map(
    db_path: str | Path | None = None,
    dictionary: MmcifDictionary | None = None,
) -> DictionarySchemaMap:
    """Build dictionary-item to MFDB table/column mappings."""
    return DictionarySchemaMap(db_path=db_path, dictionary=dictionary)


def map_dictionary_item(
    full_name: str,
    db_path: str | Path | None = None,
) -> MappedColumn | None:
    """Return the mapped MFDB table/column for one dictionary item."""
    return build_dictionary_schema_map(db_path).map_dictionary_item(full_name)


def unsupported_flr_items(db_path: str | Path) -> list[str]:
    """Return flrCIF items that do not map to the given live schema."""
    mapper = build_dictionary_schema_map(db_path)
    return [item.dictionary_name for item in mapper.get_unmapped_flr_items()]
