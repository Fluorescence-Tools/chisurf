---
type: Plugin Group
title: Sample, Spectra & Curation
description: Database and calibration-oriented tools for samples, spectra, PCH analysis, Jordi files, and AI/provider settings.
resource: chisurf/plugins/
tags: [plugins, mfdb, spectra, pch, calibration]
timestamp: '2026-07-05T00:00:00Z'
---

This group covers cross-cutting scientific utilities that do not fit cleanly
into one acquisition modality: sample/provenance databases, optical-component
spectra, photon-counting histograms, Jordi calibration files, and AI/provider
configuration.

| Plugin dir | Display name | What it does |
| --- | --- | --- |
| `sample_database` | Legacy:Sample Database | Retired prerelease MFDB surface; active work belongs in `core/mfdb_admin` and canonical `mfdb.*` services. |
| `spectra_downloader` | Spectra Downloader | Downloads, browses, stages, and pushes fluorophore/filter/dichroic/detector/light-source spectra to MFDB endpoints. |
| `pch` | Spectroscopy:Single-Molecule:PCH | Computes photon-counting histograms from TTTR files and fits multi-species brightness/occupancy models. |
| `jordi_g_factor` | Spectroscopy:Fluorescence decay:Jordi G-Factor Calculator | Calculates detector G-factors from Jordi-format decay files, with CLI and backend services. |
| `jordi_anisotropy` | Spectroscopy:Fluorescence decay:Jordi Anisotropy Decay | Computes anisotropy decays from Jordi files, including batch processing. |
| `ai_settings` | Tools:AI Settings | Configures API providers and backends for AI-assisted features. |

`sample_database` is legacy. ChiSurf is prerelease, so compatibility is not required:
ongoing work should converge on MFDB admin, project-browser, and object-store paths
and remove old `sample_database.*` aliases instead of maintaining duplicate database
frontends. PCH has a clean GUI/CLI/service split (`pch.load_tttr`, `pch.compute`,
`pch.fit`) and is a good reference for keeping numerical code in `api/` with UI in
`gui/`.

See also [MFDB](/architecture/mfdb.md), [fluorescence domain](/subsystems/fluorescence-domain.md),
[TTTR tools](/plugins/tttr.md), and [fluorescence decay](/plugins/fluorescence-decay.md).
