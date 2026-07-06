# MFDB vendored package

MFDB is the metadata/provenance database package used by ChiSurf.

The package is vendored here during prerelease extraction so ChiSurf can start
depending on `mfdb` as a separate module before the code is moved to
`github.com/fluorescence-tools/mfdb`.

Current state:

- Canonical source package: `modules/mfdb/src/mfdb`
- Transitional compatibility import: `chisurf.core.mfdb`
- Target import for new code: `mfdb`
- Runtime path/default-user configuration: `mfdb.config`, with `MFDB_*`
  environment variables for ChiSurf embedding and standalone tests.
- Remaining extraction debt: remove ChiSurf-specific imports from the vendored
  package and move the MFDB Admin service/frontend boundary into the package or
  a companion plugin.

This directory is intentionally package-shaped, not a documentation-only
placeholder.
