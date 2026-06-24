"""Tests for the Help plugin API layer."""

from chisurf.plugins.core.help.api.io import discover_docs
from chisurf.plugins.core.help.api.markdown import render_markdown
from chisurf.plugins.core.help.gui.client import HelpClient


def test_discover_docs_returns_entries():
    info = discover_docs()
    assert info.entries
    assert info.tree


def test_render_markdown_adds_heading_id():
    html = render_markdown("# Hello {#world}\n\nContent")
    assert "id=\"world\"" in html
    assert "Content" in html


def test_help_client_can_list_docs():
    client = HelpClient()
    result = client.list_docs()
    assert result.get("ok") is True
    payload = result.get("result", {})
    assert "entries" in payload
    assert "tree" in payload
    assert payload["entries"]


def test_help_client_can_search_docs():
    client = HelpClient()
    results = client.search_docs("fcs")
    assert results
    assert {"path", "title", "match_type"} <= set(results[0])


def test_help_client_can_read_doc():
    info = discover_docs()
    path = info.entries[0].path
    client = HelpClient()
    result = client.read_doc(path)
    assert result is not None
    assert "content" in result
    assert "html" in result
