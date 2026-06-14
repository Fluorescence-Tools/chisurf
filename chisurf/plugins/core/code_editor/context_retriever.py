from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chisurf.plugins.core.code_editor.wiki_indexer import build_api_index

_WORD_RE = re.compile(r"[a-z0-9_]{3,}", re.IGNORECASE)
_IMPORT_RE = re.compile(r"^(?:from\s+(?P<from>[a-zA-Z0-9_.]+)\s+import\s+|import\s+)(?P<name>[a-zA-Z0-9_.]+)")


class ChisurfContextRetriever:
    """Retrieve verified ChiSurf API context for the agent."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        """Initialize the retriever.

        Parameters
        ----------
        repo_root : str or pathlib.Path, optional
            Repository root. Defaults to the parent of the ``chisurf`` package.
        """
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[4]).resolve()
        self.index_path = self.repo_root / "llm-wiki" / "wiki" / "api" / "index.json"

    def retrieve_context(
        self,
        query: str,
        current_file_content: str = "",
        limit: int = 8,
    ) -> str:
        """Return verified ChiSurf API context for a user query.

        Parameters
        ----------
        query : str
            User request text.
        current_file_content : str, optional
            Current editor content used to bias retrieval toward imported symbols.
        limit : int, optional
            Maximum number of API symbols to include.

        Returns
        -------
        str
            Markdown-formatted API context.
        """
        symbols = self._load_symbols()
        if not symbols:
            return ""
        ranked = self._rank_symbols(query, current_file_content, symbols)
        if not ranked:
            return ""
        return self._format_context(ranked[:limit])

    def _load_symbols(self) -> list[dict[str, Any]]:
        """Load the API index, building it lazily if it is missing."""
        if not self.index_path.is_file():
            build_api_index(self.repo_root)
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        symbols = payload.get("symbols", [])
        return [symbol for symbol in symbols if isinstance(symbol, dict)]

    def _rank_symbols(
        self,
        query: str,
        current_file_content: str,
        symbols: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank API symbols by relevance to the query and current file."""
        query_terms = _WORD_RE.findall(query.lower())
        current_imports = self._current_imports(current_file_content)
        scored: list[tuple[int, dict[str, Any]]] = []
        for symbol in symbols:
            score = self._score_symbol(symbol, query_terms, current_imports)
            if score > 0:
                scored.append((score, symbol))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [symbol for _score, symbol in scored]

    def _score_symbol(
        self,
        symbol: dict[str, Any],
        query_terms: list[str],
        current_imports: set[str],
    ) -> int:
        """Score one API symbol for a query."""
        haystack = " ".join(
            str(symbol.get(key, ""))
            for key in ["qualname", "name", "module", "path", "signature", "docstring", "source"]
        ).lower()
        score = 0
        qualname = str(symbol.get("qualname", "")).lower()
        module = str(symbol.get("module", "")).lower()
        name = str(symbol.get("name", "")).lower()
        for term in query_terms:
            if term in qualname:
                score += 20
            if term in module:
                score += 10
            if term in name:
                score += 8
            if term in haystack:
                score += 3
        for imported in current_imports:
            if imported in qualname or imported in module or imported == name:
                score += 25
        return score

    def _current_imports(self, content: str) -> set[str]:
        """Extract ChiSurf import targets from editor content."""
        imports: set[str] = set()
        for line in content.splitlines():
            match = _IMPORT_RE.match(line.strip())
            if not match:
                continue
            value = match.group("from") or match.group("name")
            if value and value.startswith("chisurf"):
                imports.add(value)
                parts = value.split(".")
                if len(parts) > 1:
                    imports.add(parts[-1])
        return imports

    def _format_context(self, symbols: list[dict[str, Any]]) -> str:
        """Format API symbols as model-ready context."""
        lines = [
            "# Verified ChiSurf API Context",
            "",
            "Use only ChiSurf APIs present below. If the requested API is not listed, say so and ask for clarification instead of inventing function names.",
            "",
        ]
        for symbol in symbols:
            lines.extend(
                [
                    f"## `{symbol.get('qualname', '')}`",
                    f"- Source: `{symbol.get('path', '')}:{symbol.get('line', '')}`",
                    f"- Kind: `{symbol.get('kind', '')}`",
                    f"- Signature: `{symbol.get('signature', '')}`",
                ]
            )
            docstring = str(symbol.get("docstring") or "").strip()
            if docstring:
                summary = " ".join(docstring.split()[:60])
                lines.append(f"- Docstring: {summary}")
            source = str(symbol.get("source") or "").strip()
            if source:
                lines.extend(["", "```python", source[:2400], "```", ""])
        return "\n".join(lines)


def retrieve_context(
    query: str,
    current_file_content: str = "",
    repo_root: str | Path | None = None,
    limit: int = 8,
) -> str:
    """Retrieve verified ChiSurf API context.

    Parameters
    ----------
    query : str
        User request text.
    current_file_content : str, optional
        Current editor content used to bias retrieval.
    repo_root : str or pathlib.Path, optional
        Repository root.
    limit : int, optional
        Maximum number of API symbols to include.

    Returns
    -------
    str
        Markdown-formatted API context.
    """
    return ChisurfContextRetriever(repo_root).retrieve_context(query, current_file_content, limit)
