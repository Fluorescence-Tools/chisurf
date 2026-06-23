# PRD-32: Acquisition Standard Output Folder

## Status

Implemented in the current branch for the configuration/runtime path. The
acquisition settings panel now exposes a standard output folder, the acquisition
dock reads it by default, and new runs resolve their output location from that
setting before starting.

## Goal

Give acquisition a single, user-configurable standard output folder so new
measurements have a predictable save location without making the user pick a
directory every time.

## Why

The current acquisition flow already has an ad-hoc output-path field in the dock,
but the setting is not defined in the central setup surface. That makes the save
location easy to miss and hard to standardize across sessions.

This PRD makes the setting explicit in the setup UI and treats it as the default
runtime destination for acquisition output.

## Scope

- Add a standard output-folder field to acquisition settings in Setup.
- Persist the setting in `gui.acquisition`.
- Prefill the acquisition dock from the saved setting.
- Resolve the output folder at acquisition start if the dock is empty.
- Create the destination folder before the run writes files.

## Non-goals

- Direct MFDB registration.
- Changing the device-specific file formats.
- Introducing a new project-scoped output tree policy.

## Definition of Done

- [ ] Acquisition settings expose a standard output-folder field.
- [ ] The folder persists across restarts.
- [ ] The acquisition dock defaults to the saved folder.
- [ ] Acquisition start uses the saved folder when the dock field is empty.
- [ ] The output folder is created before a run writes files.

## Notes

This PRD is intentionally narrow. The MFDB alternative is a separate, more
complex PRD because it needs sample ownership and provenance decisions first.
