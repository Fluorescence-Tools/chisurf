"""Qt-free view-model backing the PTU Header Editor tool.

:class:`HeaderEditorViewModel` holds the parsed PTU header tags, converts values
by tag type and writes a modified PTU through :mod:`tttrlib`. The editable tag
table lives in the custom ``header_table`` section; the read-only JSON view binds
to :attr:`json_text`. Free of Qt so it is unit-testable headlessly.
"""

from __future__ import annotations

import json
import logging
import pathlib

import tttrlib

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "header.view.json"

#: Sample header used when the tool opens without a file.
SAMPLE_JSON = json.dumps(
    {
        "tags": [
            {
                "name": "File_GUID",
                "type": 1073872895,
                "value": "{D3D2D9C0-5B48-4D94-98CE-A5E2EA3558E6}",
            },
            {"name": "File_CreatingTime", "type": 553648136, "value": "1539705731.9470003"},
            {"name": "Measurement_SubMode", "type": 268435464, "value": "1"},
            {"name": "User_Author", "type": 2000000001, "value": "John Doe"},
            {"name": "File_Version", "type": 1000000002, "value": "1.0.0"},
            {
                "name": "File_Description",
                "type": 2000000003,
                "value": "This is a sample file description.",
            },
        ]
    }
)


class HeaderEditorViewModel:
    """State + logic for the PTU Header Editor (no Qt)."""

    TYPE_MAPPING = {
        0xFFFF0008: "Empty",
        0x00000008: "Bool",
        0x10000008: "Int8",
        0x11000008: "BitSet64",
        0x12000008: "Color8",
        0x20000008: "Float8",
        0x21000008: "DateTime",
        0x2001FFFF: "Float8Array",
        0x4001FFFF: "AnsiString",
        0x4002FFFF: "WideString",
        0xFFFFFFFF: "BinaryBlob",
    }
    REVERSE_TYPE_MAPPING = {v: k for k, v in TYPE_MAPPING.items()}

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``header.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        self.opened_path: str | None = None
        self._parsed: dict = {}
        self.json_text: str = ""
        self._observers: list = []
        self.load_json(SAMPLE_JSON)

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("header observer failed", exc_info=True)

    def update(self) -> None:
        """AutoForm hook after a bound field changes (no-op; table drives state)."""

    # ── value conversion ───────────────────────────────────────────────
    def convert_value_by_type(self, value_str: str, type_str: str):
        """Convert *value_str* to the Python type implied by the tag *type_str*."""
        try:
            if type_str in ("Int8", "BitSet64"):
                return int(value_str)
            if type_str in ("Float8", "Float8Array", "DateTime"):
                return float(value_str)
            if type_str == "Bool":
                return value_str.lower() == "true"
        except ValueError:
            logger.warning("Cannot convert %r as %s", value_str, type_str)
        return value_str

    # ── tags ───────────────────────────────────────────────────────────
    @property
    def tags(self) -> list[dict]:
        """The current header tags (``name`` / ``type`` int / ``value`` / ``idx``)."""
        return self._parsed.get("tags", [])

    def load_json(self, json_str: str) -> None:
        """Parse a header JSON string and refresh the table/JSON view."""
        self._parsed = json.loads(json_str)
        self._parsed.setdefault("tags", [])
        self._refresh_json_text()
        self.notify("loaded")

    def load_ptu(self, path: str) -> None:
        """Load the header tags from the PTU file at *path*."""
        tttr = tttrlib.TTTR(path)
        self.opened_path = path
        self.load_json(tttr.header.json)

    def set_tags(self, rows: list[dict]) -> None:
        """Replace the tag list (each row: name, type-name str, value-str, idx).

        Rows arrive from the table widget with string types/values; they are
        normalised to the numeric tag type and Python value here.
        """
        tags = []
        for row in rows:
            type_str = row.get("type", "")
            type_code = self.REVERSE_TYPE_MAPPING.get(type_str)
            if type_code is None:
                continue
            tags.append(
                {
                    "name": row.get("name", ""),
                    "type": type_code,
                    "value": self.convert_value_by_type(str(row.get("value", "")), type_str),
                    "idx": int(row.get("idx", -1))
                    if str(row.get("idx", "-1")).lstrip("-").isdigit()
                    else -1,
                }
            )
        self._parsed["tags"] = tags
        self._refresh_json_text()
        self.notify("json")

    def _refresh_json_text(self) -> None:
        self.json_text = json.dumps(self._parsed, indent=4)

    # ── save ───────────────────────────────────────────────────────────
    def can_save(self) -> str | None:
        """Return ``None`` when a save can run, else a human-readable reason."""
        if not self.opened_path:
            return "Open a PTU file first (the event data is copied from it)."
        return None

    def save(self, path: str) -> None:
        """Write a new PTU at *path* with the original events and edited tags."""
        if not self.opened_path:
            raise ValueError("No source PTU file is open.")
        if not path.endswith(".ptu"):
            path += ".ptu"
        src = tttrlib.TTTR(self.opened_path)
        out = tttrlib.TTTR()
        header_dict = json.loads(src.header.json)
        header_dict["tags"] = self._parsed.get("tags", [])
        out.header.set_json(json.dumps(header_dict))
        out.append_events(
            macro_times=src.macro_times,
            micro_times=src.micro_times,
            routing_channels=src.routing_channels,
            event_types=src.event_types,
        )
        out.write(path)


__all__ = ["HeaderEditorViewModel"]
