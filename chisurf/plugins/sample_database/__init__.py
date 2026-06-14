"""Legacy compatibility wrapper — delegates to chisurf.plugins.core.mfdb_admin."""

from __future__ import annotations

from chisurf.plugins.core.mfdb_admin import MFDBWidget as SampleDatabaseWidget

__all__ = ["SampleDatabaseWidget"]
