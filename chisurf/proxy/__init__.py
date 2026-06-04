from __future__ import annotations

from chisurf.client import ChisurfClient
from chisurf.proxy.lists import ProxyDatasetList, ProxyFitList


def install_proxies(client: ChisurfClient) -> None:
    """Replace ``chisurf.fits``, ``chisurf.imported_datasets`` with proxy objects.

    Must be called after the server subprocess is running and the client
    is connected.  The proxies delegate all reads/writes to the server
    via ZMQ/JSON-RPC, keeping the two processes in sync.
    """
    import chisurf

    chisurf.fits = ProxyFitList(client)
    chisurf.imported_datasets = ProxyDatasetList(client)
    chisurf.__client__ = client
