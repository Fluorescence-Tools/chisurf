# ZMQ Migration Review Notes

Date: 2026-06-15

## Current Verification

- `python -m pytest test/test_forbidden_communication.py test/test_fitting_client.py -q --no-cov`
  - Result: 89 passed, 2 warnings.
- Smoke check: `FittingClient.create_fit()` reaches `fit.create` when the server reports zero existing fits.
- Smoke check: `FittingClient.sampling_status()` reaches `fit.sample.status` without a fit-list preflight.
- Smoke check: group RPC calls use `group_fit_uid` / `member_fit_uid` and no longer fail dispatcher parameter validation.

## Remaining Issues

- `FittingClient.get_fit_objects()` remains as a deprecated bridge to live `cs.fits` objects. It is intentionally exempted from the static guard while dependent widgets are migrated.
- The forbidden-communication guard currently baselines 69 direct-access violations across 19 files. The guard prevents increases but does not mean the ZMQ-only migration is complete.
- Several UI paths still read or mutate process-global fit/dataset state directly. These should be migrated to DTO/RPC APIs in follow-up work instead of expanding the baseline.

## Commit Guidance

- Keep the fitting-client/server RPC adapter changes separate from unrelated architecture migrations where possible.
- Do not remove the static guard baseline without first replacing the legacy direct-access paths.
- New fitting/widget code should use JSON-safe DTOs from `list_fits()`, `get_fit()`, or dedicated RPC endpoints, not `get_fit_objects()`.
