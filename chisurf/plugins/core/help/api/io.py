"""File I/O for the Help plugin — document discovery and access."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chisurf as cs
import chisurf.core.settings
import chisurf.plugins
from chisurf.plugins.core.help.api.markdown import extract_title


@dataclass
class DocEntry:
    """Single documentation entry from discovery."""

    path: str
    title: str
    category: str
    file_name: str
    size: int


@dataclass
class DocInfo:
    """Full document index produced by :func:`discover_docs`."""

    entries: List[DocEntry] = field(default_factory=list)
    tree: Dict[str, List[Dict]] = field(default_factory=dict)


def discover_docs() -> DocInfo:
    """Discover all Markdown documentation files.

    Returns
    -------
    DocInfo
        All discovered documents with tree structure and flat list.

    """
    entries: List[DocEntry] = []
    tree: Dict[str, List[Dict]] = {
        "User manual": [],
        "Core": [],
        "Plugins": [],
    }

    base = pathlib.Path(cs.__file__).resolve().parent
    root = base.parent

    # User manual docs
    docs_dir = root / "docs"
    if docs_dir.exists():
        for path in sorted(docs_dir.rglob("*.md")):
            try:
                rel = path.relative_to(docs_dir)
            except ValueError:
                rel = path.name
            title = _get_title(path, str(rel))
            entries.append(
                DocEntry(
                    path=str(path),
                    title=title,
                    category="User manual",
                    file_name=str(rel),
                    size=path.stat().st_size,
                )
            )
            tree["User manual"].append(
                {"path": str(path), "title": title, "file_name": str(rel)}
            )

    # Core project .md files
    for path in sorted(root.rglob("*.md")):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "docs":
            continue
        if "plugins" in rel.parts:
            continue
        title = _get_title(path, str(rel))
        entries.append(
            DocEntry(
                path=str(path),
                title=title,
                category="Core",
                file_name=str(rel),
                size=path.stat().st_size,
            )
        )
        tree["Core"].append(
            {"path": str(path), "title": title, "file_name": str(rel)}
        )

    # Plugin docs
    try:
        plugin_infos = list(cs.plugins.iter_plugins())
    except Exception:
        plugin_infos = []

    for info in plugin_infos:
        plugin_dir = pathlib.Path(info.get("package_dir")).resolve()
        markdown_files = sorted(plugin_dir.rglob("*.md"))
        if not markdown_files:
            continue

        readme_path = None
        for p in markdown_files:
            if p.name.lower() in {"readme.md", "readme"}:
                readme_path = p
                break
        if readme_path is not None:
            readme_text = readme_path.read_text(encoding="utf-8")
            readme_title = extract_title(readme_text)
        else:
            readme_title = None

        plugin_name = info.get("plugin_name") or info.get("module_name") or plugin_dir.name
        clean_name = plugin_name.split(":")[-1].strip() if ":" in plugin_name else plugin_name
        plugin_label = readme_title or clean_name or plugin_dir.name

        plugin_tree_entries = []
        for md_path in markdown_files:
            try:
                rel = md_path.relative_to(plugin_dir)
            except ValueError:
                rel = md_path.name
            title = _get_title(md_path, str(rel))
            entries.append(
                DocEntry(
                    path=str(md_path),
                    title=title,
                    category=f"Plugins/{plugin_label}",
                    file_name=str(rel),
                    size=md_path.stat().st_size,
                )
            )
            plugin_tree_entries.append(
                {"path": str(md_path), "title": title, "file_name": str(rel)}
            )

        # Use the category-style grouping
        tree_key = f"Plugins/{plugin_label}"
        tree[tree_key] = plugin_tree_entries

    return DocInfo(entries=entries, tree=tree)


def read_doc(path_str: str) -> Optional[str]:
    """Read a documentation file.

    Parameters
    ----------
    path_str : str
        Filesystem path to the Markdown file.

    Returns
    -------
    str or None
        File contents, or *None* if the file cannot be read.

    """
    path = pathlib.Path(path_str)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def save_doc(path_str: str, content: str) -> bool:
    """Save *content* to a documentation file.

    Parameters
    ----------
    path_str : str
        Filesystem path to write to.
    content : str
        New file content.

    Returns
    -------
    bool
        *True* on success.

    """
    path = pathlib.Path(path_str)
    try:
        path.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def search_docs(query: str) -> List[Dict]:
    """Search documentation files for *query*.

    Parameters
    ----------
    query : str
        Lowercased search term.

    Returns
    -------
    list of dict
        Matching entries with ``path``, ``title``, ``match_type``.

    """
    results: List[Dict] = []
    query_lower = query.lower()
    try:
        info = discover_docs()
    except Exception:
        return results

    for entry in info.entries:
        path = pathlib.Path(entry.path)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""

        lower_text = text.lower()

        if query_lower in entry.title.lower():
            match_type = "title"
        elif query_lower in entry.file_name.lower():
            match_type = "filename"
        elif query_lower in lower_text:
            match_type = "content"
        else:
            continue

        results.append(
            {
                "path": entry.path,
                "title": entry.title,
                "match_type": match_type,
            }
        )

    return results


# ── helpers ─────────────────────────────────────────────────────────


def _get_title(path: pathlib.Path, fallback: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return fallback
    title = extract_title(text)
    return title if title else fallback
