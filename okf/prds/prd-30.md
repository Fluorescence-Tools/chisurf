---
type: PRD
prd: "30"
title: "PRD-30: CLI Pipeline Tools with Unix Pipe Support"
description: Gives burst/TTTR CLI tools stdin/stdout streaming via a self-describing msgpack frame format so they compose as Unix pipes.
status: planned
phase: "cross-cutting"
resource: overhaul/PRD-30-cli-pipeline-tools.md
tags: [prd, core]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Equips every burst/TTTR CLI tool to read photon streams from stdin and write results to stdout so users can compose ad-hoc processing pipelines with Unix pipes. Because binary TTTR formats are file-position-dependent and not naively pipeable, a streamable, length-prefixed msgpack frame format (`PipeFrame`, reusing the MFDB payload codec) carries typed, self-describing packets for photon streams, burst tables, and results. Each Click command gains a positional `input` accepting a file path or `-`; no `--output` means stdout, and stderr is reserved for logging so piped data stays clean. New lean commands (`csc burst-select`, `csc photon-filter`, `csc compute-fret`, etc.) plus retrofits of existing tools make CLI tools composable building blocks.

# Status
Planned. Phased: frame format + reader/writer utilities, core pipe commands, analysis pipe commands, then retrofitting existing CLIs.

# Relationships
- Each command maps to a typed transformer per [PRD-16](prd-16.md); the burst_table payload matches [PRD-03](prd-03.md).
- A shell pipe is the ad-hoc form of the structured [PRD-22](prd-22.md) pipeline engine; complements the visual editor of [PRD-29](prd-29.md).
- Frame format adapts the streaming payload work related to [PRD-30 payload codec].
- Complemented by [PRD-31](prd-31.md), which makes the companion exploration tool a headless pipe stage.
- Relates to [server](/architecture/server.md) for network-pipe scenarios.

# Source
- Primary: `overhaul/PRD-30-cli-pipeline-tools.md`
