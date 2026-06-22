"""SQLAlchemy ORM mapping for MFDB.

This package provides a bounded SQLAlchemy mapping layer for the MFDB sample/probe/FRET
slice. It creates a deliberate relationship boundary for the tables that PRD-02a and
PRD-02 depend on, without replacing the existing MFDB repository API.

The ORM models live behind a small internal adapter; callers do not create sessions
directly. Existing callers continue using MFDatabase and high-level modules such as
sample_manager.py.
"""

from .base import make_engine, session_scope, session_from_mfdatabase
from .models import (
    FlrSample,
    FlrSampleCondition,
    FlrSampleProbe,
    FlrPolyProbePosition,
    Entity,
    EntityPolySeq,
    Probe,
    OpticalProperty,
    Spectrum,
    FlrFretForsterRadius,
    ChemDescriptor,
)
from .sample_repository import (
    create_sample_graph,
    get_sample_graph,
    upsert_probe,
)

__all__ = [
    # Base utilities
    "make_engine",
    "session_scope",
    "session_from_mfdatabase",
    # ORM models
    "FlrSample",
    "FlrSampleCondition",
    "FlrSampleProbe",
    "FlrPolyProbePosition",
    "Entity",
    "EntityPolySeq",
    "Probe",
    "OpticalProperty",
    "Spectrum",
    "FlrFretForsterRadius",
    "ChemDescriptor",
    # Repository adapter
    "create_sample_graph",
    "get_sample_graph",
    "upsert_probe",
]
