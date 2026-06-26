"""Backend (services + state) for the burst-wise FCS correlator plugin."""

from .services import register_services

__all__ = ["register_services"]
