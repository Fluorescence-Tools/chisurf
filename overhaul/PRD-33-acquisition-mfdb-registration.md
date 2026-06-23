# PRD-33: Acquisition-to-MFDB Registration

## Status

Proposed.

## Goal

Offer a second acquisition save mode: write a new measurement directly to MFDB
instead of, or in addition to, the file-system output folder.

## Why

The output-folder solution covers the immediate need, but the acquisition system
also needs a native MFDB path for users who want measurements registered as
first-class database objects as soon as they are acquired.

That path is more complex than file output because MFDB needs a provenance and
ownership answer:

- Link the measurement to an existing sample.
- Or create a new sample as part of acquisition.
- Or make the sample selection explicit at save time.

## Scope

- Add an acquisition save mode for MFDB registration.
- Let the user choose between an existing sample and a new sample.
- Register the raw measurement as an MFDB artifact.
- Preserve provenance from acquisition settings and device metadata.
- Keep file-output mode available.

## Dependencies

- PRD-02 sample tracking and sample creation.
- PRD-03 result registration.
- PRD-32 acquisition output folder, which remains the default immediate path.

## Non-goals

- Redesigning the acquisition data model.
- Collapsing measurement registration into the burst pipeline.
- Replacing the output-folder mode.

## Definition of Done

- [ ] Acquisition can register a measurement directly to MFDB.
- [ ] The user can attach the measurement to an existing sample.
- [ ] The user can create a new sample during registration.
- [ ] The MFDB record keeps the acquisition metadata and provenance.
- [ ] File output still works as the fallback path.

## Notes

This PRD should stay separate from PRD-32. The implementation path and the
database path solve different problems and have different dependency chains.
