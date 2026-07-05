---
type: Plugin Group
title: TTTR tools
description: Photon-stream tools for converting, browsing, shifting, splitting, previewing and summarizing TTTR data.
resource: chisurf/plugins/tttr/
tags: [plugins, tttr, photons, converters]
timestamp: '2026-07-05T00:00:00Z'
---

The `tttr/` group works directly on Time-Tagged Time-Resolved photon streams.
These tools are close to [data IO](/subsystems/data-io.md): most load files via
the TTTR backend, then either transform files, extract metadata/traces/images,
or compute quick diagnostics before downstream [burst](/plugins/burst.md),
[FCS](/plugins/fcs.md), [fluorescence-decay](/plugins/fluorescence-decay.md),
or [imaging](/plugins/imaging.md) analysis.

| Plugin dir | Display name | What it does |
| --- | --- | --- |
| `tttr_toolbox` | Tools:TTTR Tools | Unified toolbox for ALEX Creator, Micro-time Shifter, PTU Header Editor, and split/convert workflows. |
| `ptu_alex_creator` | Tools:Converter:ALEX Creator | Converts macro-time ALEX modulation into micro-time / PIE-style files; supports single, batch, and merged output plus histogram inspection. |
| `tttr_microtime_shifter` | Tools:TTTR:Microtime Shifter | Applies global and per-channel micro-time shifts to TTTR files with replayable transform metadata. |
| `ptu_header_edit` | Tools:TTTR:PTU Header Editor | Edits PTU header fields through a declarative view. |
| `tttr_time_windows` | Tools:Converter:TTTR->Time-Window BIDs | Splits TTTR files into fixed-duration Burst-ID windows. |
| `trace_browser` | Spectroscopy:Single-Molecule:Trace Browser | Browses folders of PTU/TTTR intensity traces, stores ratings/annotations, previews traces, and exports selections. |
| `tttr_image_browser` | Imaging:Tools:Image Browser | Previews intensity images for detector-window definitions and exports TIFF images. |
| `tttr_count_rate_analysis` | Tools:TTTR:Count Rate Analysis | Computes per-channel count rates across many TTTR files. |
| `tttr_lut_tools` | Tools:TTTR:LUT Tools | Builds micro-time lookup tables and channel-LUT settings. |
| `microtime_histogram` | Spectroscopy:Fluorescence decay:Histogram-Microtime | Creates and inspects TTTR micro-time histograms. |
| `audifier` | Tools:TTTR:Audifier | Converts photon streams to audio with a live micro-time/lifetime waterfall preview. |

Several tools expose CLI and service entrypoints in `manifest.json`, especially
the file-transforming workflows (`alex.*`, `microtime_shift.*`,
`tttr_time_windows.*`, trace/image-browser APIs). GUI state is increasingly
data-driven through AutoForm `*.view.json` files and shared toolbox shells.

See also [compiled modules](/subsystems/compiled-modules.md), [fluorescence domain](/subsystems/fluorescence-domain.md),
and [plugin system](/architecture/plugin-system.md).
