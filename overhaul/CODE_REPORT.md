# Code Report

## PRD-03 Result Registry

### Weakness: registry API arrived before plugin data interfaces are mature

The result registry now provides a single MFDB entry point for plugin outputs,
but the surrounding ChiSurf plugin layer still lacks consistent interfaces for
describing result payloads, source artifact IDs, sample IDs, and export formats.
The FCS correlator integration is therefore intentionally best-effort: it can
store the correlation payload and operation parameters, but it cannot yet attach
complete provenance when the plugin has not carried MFDB identifiers through its
workflow.

This suggests PRD-07 should not only add calls to `register_result()`. It should
first define or tighten plugin-side contracts for:

- canonical result payload formats per plugin family;
- how plugins expose `sample_id` and source `artifact_id`;
- how in-memory results map to durable files or JSON payloads;
- how metadata and fit/correlation parameters are normalized before archival.

Without those interfaces, PRD-03 is useful as a choke point, but downstream
plugin integrations will remain uneven and will often register partial
provenance.
