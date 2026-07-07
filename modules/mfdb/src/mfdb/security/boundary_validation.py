"""Dictionary-driven boundary parameter validation (PRD-26 Task 4).

The ``.dic`` is the single source of truth for the schema (PRD-19) and operation
parameter schemas (PRD-11). The repository/DAO already whitelist identifiers against
the live schema; this module adds the *value* side of the boundary: given a table (a
dictionary category) and a payload mapping, it checks the payload against the
dictionary-declared columns — rejecting unknown columns, missing mandatory ones,
type-incoherent values, and out-of-vocabulary values — so RPC/boundary requests are
dictionary-typed rather than hand-validated per handler.

It shares one type-coercion check (:func:`check_value_type`) with the PRD-11
operation-parameter validation so both boundaries agree on what ``int``/``float``/
``bool``/date values are.

Usage
-----
>>> from mfdb.security.boundary_validation import DictionaryValidator
>>> v = DictionaryValidator.load_bundled()
>>> v.validate("flr_sample", {"sample_name": "A", "sample_id": 1})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mfdb.schema.pdbx_metadata import MmcifDictionary

logger = logging.getLogger(__name__)


class BoundaryValidationError(ValueError):
    """A payload violates the dictionary-declared schema for its table."""


#: mmCIF ``type_code`` (and PRD-11 ``value_type``) values, normalised to a small set
#: of value kinds the boundary actually checks. Anything unlisted is treated as a
#: free string (``str``) — coherent, never spuriously rejected.
_INT_TYPES = frozenset({"int", "positive_int"})
_FLOAT_TYPES = frozenset({"float"})
_BOOL_TYPES = frozenset({"boolean", "bool"})
_DATE_TYPES = frozenset({"yyyy-mm-dd", "yyyy-mm-dd:hh:mm", "yyyy-mm-dd:hh:mm-flex"})


def _kind(type_code: str) -> str:
    tc = (type_code or "").strip().lower()
    if tc in _INT_TYPES:
        return "int"
    if tc in _FLOAT_TYPES:
        return "float"
    if tc in _BOOL_TYPES:
        return "bool"
    if tc in _DATE_TYPES:
        return "date"
    return "str"


def check_value_type(value: Any, type_code: str, *, field_name: str = "value") -> str | None:
    """Return an error message if ``value`` is incoherent for ``type_code``, else ``None``.

    ``None`` values always pass (nullability is enforced via the mandatory check, not
    here). Strings that parse to the target kind pass (SQLite/JSON payloads are often
    stringly typed). Unknown ``type_code`` kinds accept any value.
    """
    if value is None:
        return None
    kind = _kind(type_code)
    if kind == "int":
        if isinstance(value, bool):
            return f"{field_name}: expected integer, got boolean {value!r}"
        if isinstance(value, int):
            iv = value
        elif isinstance(value, float):
            # SQLite stores numbers as REAL, so an integer round-trips as an
            # integral float (1 -> 1.0); accept those, reject genuine fractions.
            if not value.is_integer():
                return f"{field_name}: expected integer, got {value!r}"
            iv = int(value)
        elif isinstance(value, str):
            try:
                iv = int(value.strip())
            except (TypeError, ValueError):
                return f"{field_name}: expected integer, got {value!r}"
        else:
            return f"{field_name}: expected integer, got {value!r}"
        if (type_code or "").strip().lower() == "positive_int" and iv < 0:
            return f"{field_name}: expected non-negative integer, got {iv!r}"
        return None
    if kind == "float":
        if isinstance(value, bool):
            return f"{field_name}: expected number, got boolean {value!r}"
        if isinstance(value, (int, float)):
            return None
        if isinstance(value, str):
            try:
                float(value.strip())
            except (TypeError, ValueError):
                return f"{field_name}: expected number, got {value!r}"
            return None
        return f"{field_name}: expected number, got {value!r}"
    if kind == "bool":
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value in (0, 1):
            return None
        if isinstance(value, str) and value.strip().lower() in (
            "0", "1", "true", "false", "yes", "no",
        ):
            return None
        return f"{field_name}: expected boolean, got {value!r}"
    if kind == "date":
        if isinstance(value, str) and value.strip():
            return None
        return f"{field_name}: expected date string, got {value!r}"
    return None


@dataclass(frozen=True)
class ColumnRule:
    """The dictionary-declared validation rule for one column."""

    column: str
    type_code: str = ""
    mandatory: bool = False
    enumerations: tuple[str, ...] = ()


@dataclass
class TableRules:
    """All :class:`ColumnRule` for one table (dictionary category)."""

    table: str
    columns: dict[str, ColumnRule] = field(default_factory=dict)

    @property
    def mandatory_columns(self) -> list[str]:
        return sorted(c.column for c in self.columns.values() if c.mandatory)


def _table_and_column(item) -> tuple[str, str]:
    """Apply the dictionary→schema mapping rule (PRD-19) for one item."""
    return (item.schema_table or item.category, item.schema_column or item.attribute)


class DictionaryValidator:
    """Validate boundary payloads against the dictionary-declared table schema."""

    def __init__(self, rules: Mapping[str, TableRules]):
        self._rules = dict(rules)

    @classmethod
    def from_dictionary(
        cls,
        dictionary: MmcifDictionary,
        tables: Iterable[str] | None = None,
    ) -> "DictionaryValidator":
        """Build column rules from the dictionary (optionally restricted to ``tables``)."""
        want = set(tables) if tables is not None else None
        rules: dict[str, TableRules] = {}
        for category_name in dictionary.categories():
            category = dictionary.get_category(category_name)
            if category is None:
                continue
            for item in category.items.values():
                table, column = _table_and_column(item)
                if want is not None and table not in want:
                    continue
                table_rules = rules.setdefault(table, TableRules(table=table))
                # First declaration wins; merge mandatory/enum on re-declaration.
                existing = table_rules.columns.get(column)
                enums = tuple(item.enumerations or ())
                if existing is None:
                    table_rules.columns[column] = ColumnRule(
                        column=column,
                        type_code=item.type_code or "",
                        mandatory=bool(item.mandatory),
                        enumerations=enums,
                    )
                else:
                    table_rules.columns[column] = ColumnRule(
                        column=column,
                        type_code=existing.type_code or (item.type_code or ""),
                        mandatory=existing.mandatory or bool(item.mandatory),
                        enumerations=existing.enumerations or enums,
                    )
        return cls(rules)

    @classmethod
    def load_bundled(cls, tables: Iterable[str] | None = None) -> "DictionaryValidator":
        """Build a validator from the bundled flrCIF + ``mfdb_flr_ext.dic`` dictionary."""
        return cls.from_dictionary(MmcifDictionary.load_bundled(), tables=tables)

    def has_table(self, table: str) -> bool:
        return table in self._rules

    def table_rules(self, table: str) -> TableRules | None:
        return self._rules.get(table)

    def validate(
        self,
        table: str,
        values: Mapping[str, Any] | None,
        *,
        require_mandatory: bool = True,
        allow_unknown: bool = False,
    ) -> None:
        """Validate ``values`` against the dictionary schema for ``table``.

        Checks (in order): unknown columns (unless ``allow_unknown``), missing
        mandatory columns (unless ``require_mandatory`` is off — e.g. partial
        updates), value type coherence, and controlled-vocabulary membership.
        No-op for a table the dictionary does not declare (backward compatible).
        Raises :class:`BoundaryValidationError` on the first violation found,
        reporting every offending field.
        """
        rules = self._rules.get(table)
        if rules is None:
            return
        provided = dict(values or {})

        if not allow_unknown:
            unknown = sorted(set(provided) - set(rules.columns))
            if unknown:
                raise BoundaryValidationError(
                    f"Unknown column(s) {unknown} for table {table!r}; "
                    f"declared: {sorted(rules.columns)}"
                )

        if require_mandatory:
            missing = sorted(
                col for col in rules.mandatory_columns
                if provided.get(col) is None
            )
            if missing:
                raise BoundaryValidationError(
                    f"Missing required column(s) {missing} for table {table!r}"
                )

        errors: list[str] = []
        for name, value in provided.items():
            rule = rules.columns.get(name)
            if rule is None:
                continue
            type_error = check_value_type(value, rule.type_code, field_name=name)
            if type_error:
                errors.append(type_error)
                continue
            if value is not None and rule.enumerations and str(value) not in rule.enumerations:
                errors.append(
                    f"{name}: {value!r} not in allowed values "
                    f"{list(rule.enumerations)}"
                )
        if errors:
            raise BoundaryValidationError(
                f"Invalid value(s) for table {table!r}: " + "; ".join(errors)
            )
