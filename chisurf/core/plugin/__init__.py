"""Plugin infrastructure: manifests, discovery, registry, client."""
from chisurf.core.plugin.client import InProcessClient, PluginClient
from chisurf.core.plugin.manifest import (
    PluginManifest,
    PluginStatefulness,
    PluginWindowState,
    load_manifest,
    validate_manifest,
)
from chisurf.core.plugin.registry import PluginRegistry

__all__ = [
    "PluginManifest",
    "PluginStatefulness",
    "PluginWindowState",
    "load_manifest",
    "validate_manifest",
    "PluginRegistry",
    "PluginClient",
    "InProcessClient",
]
