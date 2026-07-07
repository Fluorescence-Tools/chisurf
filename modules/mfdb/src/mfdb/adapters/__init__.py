"""Adapters bridging MFDB to external systems and host applications.

Each adapter maps MFDB's provenance/metadata model to (or from) another system:
``chinet`` bridges ChiSurf/chinet fit sessions into MFDB artifacts; future
adapters target electronic lab notebooks (e.g. eLabFTW) and other stores.

Adapters may depend on their target system, but must keep those imports lazy
(function-local) so the core package still imports with only ``src`` on the path.
"""
