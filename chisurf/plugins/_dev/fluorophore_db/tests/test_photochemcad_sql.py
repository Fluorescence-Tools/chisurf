"""The mysqldump parser behind the web PhotochemCAD downloader.

PhotochemCAD is fetched as MySQL dumps; ``_parse_insert_rows`` must handle the
quirks that previously broke the import: spaces after commas, NULLs, numeric vs
string values, and escaped quotes. No network — the SQL is synthetic.
"""
from __future__ import annotations

from chisurf.plugins.spectra_downloader.download.photochemcad_common_compounds import (
    _parse_insert_rows,
)


def test_parse_records_handles_spaces_nulls_and_escapes():
    sql = (
        "CREATE TABLE `records` ( `id` varchar, `name` varchar, `qy` float );\n"
        "INSERT INTO `records` (`id`, `name`, `qy`) VALUES "
        "('A01', 'Benzene', 0.053), ('A02', 'it\\'s a dye', NULL), "
        "('A03', 'comma, inside', 1.0);\n"
    )
    cols, rows = _parse_insert_rows(sql, "records")
    assert cols == ["id", "name", "qy"]
    assert len(rows) == 3
    # leading space after comma must NOT pollute the value (the join-breaking bug)
    assert rows[0] == ["A01", "Benzene", "0.053"]
    # backslash-escaped apostrophe inside a quoted string
    assert rows[1][1] == "it's a dye"
    # NULL → None
    assert rows[1][2] is None
    # comma inside a quoted string is not a field separator
    assert rows[2][1] == "comma, inside"


def test_parse_graphic_data_rows():
    sql = (
        "INSERT INTO `graphic_data` (`compound`, `wavelength`, `abs`, `ems`) VALUES "
        "('Benzene','220','10.019',NULL),('Benzene','221',NULL,'0.5');\n"
    )
    cols, rows = _parse_insert_rows(sql, "graphic_data")
    assert cols == ["compound", "wavelength", "abs", "ems"]
    assert rows[0] == ["Benzene", "220", "10.019", None]
    assert rows[1] == ["Benzene", "221", None, "0.5"]
