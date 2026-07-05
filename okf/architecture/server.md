---
type: Architecture
title: Headless Server
description: The Qt-free ZMQ/JSON-RPC 2.0 server under chisurf/server/.
resource: chisurf/server/
tags: [server, zmq, jsonrpc, headless]
timestamp: '2026-07-05T00:00:00Z'
---

# Purpose

ChiSurf includes a headless server based on ZMQ and JSON-RPC 2.0. It is
Qt-free and lives under `chisurf/server/`. In `server` mode the
[API facade](/architecture/api-facade.md) routes calls to it through
`ChisurfClient`.

# Layout

| Path | Role |
|------|------|
| `app.py` | `ChiSurfServer`, server wiring and lifecycle |
| `__main__.py` | `python -m chisurf.server` entry point |
| `startup.py` | Subprocess startup/termination helpers |
| `dispatcher.py` | `ServiceDispatcher`, method registration and invocation |
| `server_methods.json` | Declarative server RPC registry |

Run it headlessly with `python -m chisurf.server`.

# Citations

[1] [ChiSurf architecture doc](/references/architecture-doc.md)
