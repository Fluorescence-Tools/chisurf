"""RPC client for the TTTR Image Browser GUI."""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient
from chisurf.plugins.tttr.tttr_image_browser.api.contract import (
    METHOD_CONTRACT,
    METHOD_EXPORT_TIFF,
    METHOD_GET_METADATA,
    METHOD_LIST_FILES,
    METHOD_LOAD_IMAGE,
    METHOD_SET_METADATA,
)


class TTTRImageBrowserClient:
    """Typed RPC client for TTTR Image Browser."""

    def __init__(self, client: Any | None = None):
        """Wrap *client* or create a local in-process client.

        Parameters
        ----------
        client : Any, optional
            A backend client. If None, makes a local client.
        """
        self._client = client or self._make_local_client()

    def list_files(self, folder: str, recursive: bool = False, setup_settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """List image files through RPC.

        Parameters
        ----------
        folder : str
            The folder directory path.
        recursive : bool, optional
            Whether to scan recursively, by default False.
        setup_settings : dict, optional
            Wizard setup settings, by default None.

        Returns
        -------
        list of dict
            Details of matching files.
        """
        result = self._call(
            METHOD_LIST_FILES,
            {
                "folder": folder,
                "recursive": recursive,
                "setup_settings": setup_settings,
            },
        )
        return result.get("files", []) if result else []

    def get_metadata(self, folder: str) -> dict[str, dict[str, Any]]:
        """Load metadata through RPC.

        Parameters
        ----------
        folder : str
            The folder directory path.

        Returns
        -------
        dict[str, dict[str, Any]]
            Metadata dict.
        """
        result = self._call(METHOD_GET_METADATA, {"folder": folder})
        return result.get("metadata", {}) if result else {}

    def set_metadata(self, folder: str, metadata: dict[str, dict[str, Any]]) -> bool:
        """Save metadata through RPC.

        Parameters
        ----------
        folder : str
            The folder directory path.
        metadata : dict
            Metadata to save.

        Returns
        -------
        bool
            True if successful.
        """
        result = self._call(METHOD_SET_METADATA, {"folder": folder, "metadata": metadata})
        return bool(result and result.get("folder"))

    def load_image(
        self,
        path: str,
        setup_settings: dict[str, Any] | None = None,
        max_side: int = 512,
        cache_folder: str | None = None
    ) -> dict[str, Any] | None:
        """Load or precompute image mosaic through RPC.

        Parameters
        ----------
        path : str
            The image file path.
        setup_settings : dict, optional
            Wizard setup settings, by default None.
        max_side : int, optional
            Max side for scaling, by default 512.
        cache_folder : str, optional
            Custom cache folder path, by default None.

        Returns
        -------
        dict or None
            Result dict with mosaic data, or None.
        """
        result = self._call(
            METHOD_LOAD_IMAGE,
            {
                "path": path,
                "setup_settings": setup_settings,
                "max_side": max_side,
                "cache_folder": cache_folder,
            },
        )
        return result.get("image") if result else None

    def export_tiff(
        self,
        paths: list[str],
        output_dir: str,
        setup_settings: dict[str, Any] | None = None
    ) -> list[str]:
        """Export image stacks to TIFF stacks through RPC.

        Parameters
        ----------
        paths : list of str
            Input paths.
        output_dir : str
            Output directory path.
        setup_settings : dict, optional
            Wizard setup settings, by default None.

        Returns
        -------
        list of str
            List of exported TIFF paths.
        """
        result = self._call(
            METHOD_EXPORT_TIFF,
            {
                "paths": paths,
                "output_dir": output_dir,
                "setup_settings": setup_settings,
            },
        )
        return result.get("paths", []) if result else []

    def describe_contract(self) -> dict[str, Any]:
        """Return the TTTR Image Browser RPC contract.

        Returns
        -------
        dict
            RPC contract dict.
        """
        result = self._call(METHOD_CONTRACT)
        return result or {}

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        try:
            result = self._client.call(method, params)
            if result and result.get("ok"):
                return result.get("result")
        except Exception:
            return None
        return None

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local client with TTTR Image Browser services registered."""
        from chisurf.plugins.tttr.tttr_image_browser.backend.services import register_services
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()
        register_services(dispatcher)
        return InProcessClient(dispatcher)
