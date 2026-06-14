"""Legacy compatibility client for sample-database RPC services."""

from chisurf.core.plugin.client import InProcessClient
from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient


class SampleDatabaseClient(MFDBClient):
    """Client for legacy sample-database handlers.

    Examples
    --------
    >>> client = SampleDatabaseClient()
    >>> isinstance(client, SampleDatabaseClient)
    True
    """

    def __init__(self, *args, **kwargs):
        """Create a legacy in-process sample database client."""
        super().__init__(*args, **kwargs, inprocess=True)

    def _make_inprocess_client(self) -> InProcessClient:
        """Create an in-process client with legacy handlers overlaid."""
        from chisurf.plugins.core.mfdb_admin.backend.services import register_services
        from chisurf.plugins.sample_database.backend.measurement_services import (
            register_measurement_services,
        )
        from chisurf.plugins.sample_database.backend.ndxplorer_services import (
            register_ndxplorer_services,
        )
        from chisurf.plugins.sample_database.backend.setup_services import (
            register_setup_services,
        )
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        dispatcher = ServiceDispatcher(SessionState())
        register_services(dispatcher)
        register_measurement_services(dispatcher)
        register_ndxplorer_services(dispatcher)
        register_setup_services(dispatcher)
        return InProcessClient(dispatcher)


__all__ = ["SampleDatabaseClient"]
