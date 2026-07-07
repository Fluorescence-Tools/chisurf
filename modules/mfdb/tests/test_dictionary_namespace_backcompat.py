"""Back-compat for the vendor-neutral schema namespace rename (PRD-44).

The local extension tags were renamed ``_chisurf_schema.*`` -> ``_mfdb_schema.*``.
The parser keeps a legacy-fallback branch so old ``.dic`` copies (and any
third-party dictionaries) using the branded tag still parse. This guards that
fallback: a fragment written with the *legacy* tags must populate the same
``DictItem.schema_*`` fields as the new spelling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mfdb.schema.pdbx_metadata import MmcifDictionary

_LEGACY_FRAGMENT = """\
data_legacy_test

save__test_legacy_cat.value
   _item.name                "_test_legacy_cat.value"
   _item.category_id         test_legacy_cat
   _item_type.code           float
   _chisurf_schema.table_name  test_legacy_table
   _chisurf_schema.column_name legacy_value
   _chisurf_schema.status      active
   _item_description.description
;     legacy schema-mapped value
;

save__test_legacy_cat.parent_id
   _item.name                "_test_legacy_cat.parent_id"
   _item.category_id         test_legacy_cat
   _item_type.code           int
   _chisurf_schema.table_name  test_legacy_table
   _chisurf_schema.column_name parent_id
   _chisurf_schema.foreign_key parent_table(parent_id)
   _item_description.description
;     legacy schema-mapped foreign key
;
"""


def _parse_fragment(text: str) -> MmcifDictionary:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "legacy.dic"
        p.write_text(text, encoding="utf-8")
        return MmcifDictionary(p)


def test_legacy_chisurf_schema_tags_still_parse() -> None:
    dic = _parse_fragment(_LEGACY_FRAGMENT)

    value = dic.get_item("_test_legacy_cat.value")
    assert value is not None, "legacy item not parsed"
    assert value.schema_table == "test_legacy_table"
    assert value.schema_column == "legacy_value"
    assert value.schema_status == "active"

    fk = dic.get_item("_test_legacy_cat.parent_id")
    assert fk is not None
    assert fk.schema_foreign_key == "parent_table(parent_id)"


def test_new_mfdb_schema_tags_parse_identically() -> None:
    """The new spelling populates the same fields (parity with the legacy path)."""
    new_fragment = _LEGACY_FRAGMENT.replace("_chisurf_schema", "_mfdb_schema")
    dic = _parse_fragment(new_fragment)

    value = dic.get_item("_test_legacy_cat.value")
    assert value is not None
    assert value.schema_table == "test_legacy_table"
    assert value.schema_column == "legacy_value"
    assert value.schema_status == "active"
