"""GUI-side client for the central detector-setup RPC service.

Thin re-export of the shared
:class:`~chisurf.gui.widgets.wizard.tttr_channeldefinition.setup_client.DetectorSetupClient`,
which the Imaging Tools and Burst Analysis coordinators both use to talk to the
``detector_setups.*`` RPC store.
"""

from __future__ import annotations

from chisurf.gui.widgets.wizard.tttr_channeldefinition.setup_client import (
    DetectorSetupClient,
)

__all__ = ["DetectorSetupClient"]
