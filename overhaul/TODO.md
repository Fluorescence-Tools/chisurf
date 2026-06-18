# MFDB Overhaul TODO

## Deferred architecture cleanup

- Move MFDB out of `chisurf` into `modules/` as an independent Python package,
  similar to `chinet`. The package should own the MFDB schema, repository/API
  layer, MFDB server, and mfdb-admin implementation. Do this after the current
  overhaul is basically finished so the extraction can preserve the stabilized
  interfaces instead of moving churn.
