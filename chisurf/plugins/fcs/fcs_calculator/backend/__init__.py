"""Backend (services + state) for the FCS confocal calculator plugin."""

from .services import register_services

__all__ = ["register_services"]
