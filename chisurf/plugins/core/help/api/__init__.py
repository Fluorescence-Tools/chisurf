"""Help plugin API layer — pure Python, no Qt dependencies."""

from chisurf.plugins.core.help.api.contract import contract_descriptor
from chisurf.plugins.core.help.api.io import (
    DocEntry,
    DocInfo,
    discover_docs,
    read_doc,
    save_doc,
)
from chisurf.plugins.core.help.api.markdown import (
    extract_title,
    render_markdown,
    slugify_heading,
)
from chisurf.plugins.core.help.api.models import (
    HelpRequest,
    HelpResponse,
    HelpState,
)

__all__ = [
    "contract_descriptor",
    "discover_docs",
    "read_doc",
    "save_doc",
    "DocInfo",
    "DocEntry",
    "render_markdown",
    "extract_title",
    "slugify_heading",
    "HelpState",
    "HelpRequest",
    "HelpResponse",
]
