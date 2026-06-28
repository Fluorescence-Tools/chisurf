"""Derive FieldSpec definitions from the mmCIF dictionary.

This replaces the hardcoded SCHEMAS dict in generic_form.py.
Every FieldSpec is derived purely from the dictionary - no hardcoded
field/column lists allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from chisurf.core.mfdb.dictionary_schema_map import DictionarySchemaMap
from chisurf.core.mfdb.pdbx_metadata import DictItem, MmcifDictionary


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single form field, derived from the dictionary.

    Attributes
    ----------
    name : str
        Live DB column / record key (from schema map or item attribute).
    label : str
        Prettified display label (e.g. ``"sample_condition_id"`` -> ``"Sample Condition"``).
    widget : str
        Widget type: ``"str"`` | ``"text"`` | ``"int"`` | ``"float"`` | ``"bool"`` | ``"choice"``.
    choices : list[str]
        Enumeration values for choice widgets.
    required : bool
        Whether the field is mandatory.
    readonly : bool
        True for id/uuid/created_at/updated_at keys.
    default : str
        Default value from dictionary.
    tooltip : str
        Field description from dictionary.
    fk_target : str | None
        Target entity key if this is a foreign key.
    placeholder : str
        Placeholder hint text.
    """
    name: str
    label: str = ""
    widget: str = "str"
    choices: list[str] = field(default_factory=list)
    required: bool = False
    readonly: bool = False
    default: str = ""
    tooltip: str = ""
    fk_target: str | None = None
    placeholder: str = ""


# Type-code -> widget mapping. This is the only type-semantics mapping
# allowed; it maps dictionary type codes to GUI widget types.
TYPE_CODE_WIDGET_MAP: dict[str, str] = {
    "int": "int",
    "uint": "int",
    "integer": "int",
    "float": "float",
    "double": "float",
    "num": "float",
    "text": "text",
    "line": "str",
    "code": "choice",
    "char": "str",
    "ucode": "str",
    "date": "str",
    "datetime": "str",
    "yyyy-mm-dd": "str",
    "boolean": "bool",
    "yes_no": "bool",
    "enum": "choice",
}

# Attribute names that are always readonly (exact match)
READONLY_ATTRIBUTES = frozenset({
    "created_at", "updated_at",
    "object_uuid", "content_md5", "checksum",
    "storage_path", "size_bytes", "refcount", "mime_type",
})

# Attribute suffixes that make a field readonly (e.g. "created_at" -> readonly)
READONLY_SUFFIXES = frozenset({
    "uuid", "_uuid",
})


def _prettify_label(attr: str) -> str:
    """Convert snake_case attribute name to a display label."""
    if not attr:
        return ""
    s = attr.replace("_", " ")
    return s.title() if s else s


def _infer_fk_target(
    item: DictItem,
    registry: dict[str, Any] | None = None,
    id_field: str | None = None,
) -> str | None:
    """Infer FK target entity key from item parent or _id naming convention.

    A field is considered an FK only if it has an explicit ``parent`` link
    in the dictionary, or if its name ends in ``_id`` but is **not** the
    entity's own primary key field.
    """
    # Skip if this is the entity's own primary key
    attr = item.attribute or ""
    if id_field and (attr == id_field or attr == id_field.replace("_", "")):
        return None

    if item.parent:
        parent_name = item.parent
        if parent_name.startswith("_"):
            parent_name = parent_name[1:]
        parts = parent_name.split(".")
        if len(parts) >= 2:
            category = parts[0]
            # Try registry first
            if registry:
                for key, spec in registry.items():
                    if spec.get("category") == category:
                        return key
            # Fallback: derive from category name
            cat = category
            if cat.startswith("flr_"):
                cat = cat[4:]
            # Strip common suffixes
            for suffix in ("_list", "_details", "_position", "_descriptor"):
                if cat.endswith(suffix):
                    cat = cat[: -len(suffix)]
            return cat

    # Convention: column ends in _id (but not the entity's own id)
    if attr.endswith("_id") and attr not in ("id",):
        base = attr[:-3]
        return base
    return None


def field_specs_for_category(
    dic: MmcifDictionary,
    category: str,
    schema_map: DictionarySchemaMap | None = None,
    registry: dict[str, Any] | None = None,
    id_field: str | None = None,
) -> list[FieldSpec]:
    """Derive FieldSpecs for all items in a dictionary category.

    Parameters
    ----------
    dic : MmcifDictionary
        The loaded dictionary.
    category : str
        Dictionary category name (e.g. ``"flr_sample"``).
    schema_map : DictionarySchemaMap, optional
        Schema map for resolving live column names.
    registry : dict, optional
        Entity registry for FK target resolution.
    id_field : str, optional
        Override for the id field name (default: auto-detect).

    Returns
    -------
    list[FieldSpec]
        Field specifications for the category.
    """
    cat = dic.get_category(category)
    if cat is None:
        return []

    specs: list[FieldSpec] = []
    for attr, item in cat.items.items():
        spec = _field_spec_from_item(item, dic, schema_map, registry, id_field)
        specs.append(spec)

    return specs


def _field_spec_from_item(
    item: DictItem,
    dic: MmcifDictionary,
    schema_map: DictionarySchemaMap | None = None,
    registry: dict[str, Any] | None = None,
    id_field: str | None = None,
) -> FieldSpec:
    """Build a single FieldSpec from a DictItem."""
    # Resolve record key
    name = item.attribute or ""
    if schema_map:
        mapping = schema_map.map_dictionary_item(item.name)
        if mapping and mapping.column_name:
            name = mapping.column_name
    if not name:
        name = item.attribute or ""

    # Prettify label
    label = _prettify_label(name)

    # Widget type
    type_code = (item.type_code or "").lower()
    has_enums = bool(item.enumerations)
    if has_enums:
        widget = "choice"
    else:
        widget = TYPE_CODE_WIDGET_MAP.get(type_code, "str")

    # FK target (detect before readonly check)
    fk_target = _infer_fk_target(item, registry, id_field=id_field)

    # Readonly: PK fields and timestamp/audit fields
    attr_lower = name.lower()
    readonly = False
    if id_field and (attr_lower == id_field.lower() or attr_lower == id_field.lower().replace("_", "")):
        readonly = True
    if attr_lower in READONLY_ATTRIBUTES:
        readonly = True
    for suffix in READONLY_SUFFIXES:
        if attr_lower.endswith(suffix):
            readonly = True
            break

    # Required: mandatory in dictionary, but relax for FK fields
    # (FK fields are typically optional references)
    required = item.mandatory
    if fk_target is not None:
        required = False
    if readonly:
        required = False

    return FieldSpec(
        name=name,
        label=label,
        widget=widget,
        choices=list(item.enumerations),
        required=required,
        readonly=readonly,
        default=item.default_value,
        tooltip=item.description,
        fk_target=fk_target,
        placeholder=f"Enter {label.lower()}",
    )
