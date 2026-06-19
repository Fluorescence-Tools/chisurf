"""API package for the FPS JSON Editor plugin."""

from .client import FpsJsonEditorClient
from .contract import (
    CONTRACT_VERSION,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FETCH_PDB,
    METHOD_NORMALIZE_PAYLOAD,
    METHOD_SAVE_AV_MRC,
    METHOD_SUMMARIZE_PAYLOAD,
    METHOD_VALIDATE_PAYLOAD,
    PLUGIN_ID,
    contract_descriptor,
    normalize_pdb_id,
    service_success,
)

__all__ = [
    "CONTRACT_VERSION",
    "FpsJsonEditorClient",
    "METHOD_DESCRIBE_CONTRACT",
    "METHOD_FETCH_PDB",
    "METHOD_NORMALIZE_PAYLOAD",
    "METHOD_SAVE_AV_MRC",
    "METHOD_SUMMARIZE_PAYLOAD",
    "METHOD_VALIDATE_PAYLOAD",
    "PLUGIN_ID",
    "contract_descriptor",
    "normalize_pdb_id",
    "service_success",
]
