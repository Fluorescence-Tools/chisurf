"""HelpClient — typed convenience wrapper around PluginClient."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from chisurf.core.plugin.client import InProcessClient
from chisurf.plugins.core.help.api.contract import (
    METHOD_CONTRACT,
    METHOD_LIST_DOCS,
    METHOD_READ_DOC,
    METHOD_SAVE_DOC,
    METHOD_SEARCH_DOCS,
)


class HelpClient:
    """Typed RPC client for the Help plugin.

    Wraps a ``PluginClient`` (either ``InProcessClient`` or ``ZmqClient``)
    and provides convenience methods for each ``help.docs.*`` endpoint.
    """

    def __init__(self, client: Optional[Any] = None):
        """Wrap *client* (a PluginClient-compatible object).

        Parameters
        ----------
        client : PluginClient, optional
            The transport client (InProcessClient or ZmqClient). If omitted,
            creates a local in-process client with the Help services registered.

        """
        if client is None:
            self._client = self._make_local_client()
        else:
            self._client = client

    def list_docs(self) -> Dict[str, Any]:
        """List all available documentation files.

        Returns
        -------
        dict
            ``{"entries": [...], "tree": {...}}``

        """
        return self._call(METHOD_LIST_DOCS)

    def read_doc(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a documentation file.

        Parameters
        ----------
        path : str
            Filesystem path to the Markdown file.

        Returns
        -------
        dict or None
            ``{"content": str, "html": str, "title": str}`` or *None*.

        """
        result = self._call(METHOD_READ_DOC, {"path": path})
        if result and result.get("ok"):
            return result.get("result")
        return None

    def save_doc(self, path: str, content: str) -> bool:
        """Save content to a documentation file.

        Parameters
        ----------
        path : str
            Filesystem path to write to.
        content : str
            New file content.

        Returns
        -------
        bool
            *True* on success.

        """
        result = self._call(METHOD_SAVE_DOC, {"path": path, "content": content})
        return bool(result and result.get("ok"))

    def search_docs(self, query: str) -> List[Dict[str, Any]]:
        """Search documentation files.

        Parameters
        ----------
        query : str
            Search term.

        Returns
        -------
        list of dict
            Matching entries.

        """
        result = self._call(METHOD_SEARCH_DOCS, {"query": query})
        if result and result.get("ok"):
            return result.get("result", [])
        return []

    def describe_contract(self) -> Dict[str, Any]:
        """Return the Help plugin workflow contract."""
        result = self._call(METHOD_CONTRACT)
        if result and result.get("ok"):
            return result.get("result", {})
        return {}

    def _call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            return self._client.call(method, params)
        except Exception:
            return None

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with Help services."""
        from chisurf.plugins.core.help.backend.services import register_services
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()
        register_services(dispatcher)
        return InProcessClient(dispatcher)
