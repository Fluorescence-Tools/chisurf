---
type: Plugin Profile
title: Code Editor plugin
description: OKF profile for the shared code/text editor plugin.
resource: chisurf/plugins/core/code_editor/
tags: [plugins, editor, rpc, tooling]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `code_editor` |
| Display name | `Tools:Miscellaneous:Code Editor` |
| Categories | `Tools`, `Miscellaneous` |
| Version | `2.1.0` |
| State namespace | `code_editor` |
| Local README | Missing |

The manifest describes a shared multi-document editor with project navigation,
symbols, diagnostics, and optional Python LSP integration.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| GUI/editor | `window.py`, `editor.py`, `text_editor.py`, `agent_panel.py`. |
| Tooling | `lsp_client.py`, `ruff_runner.py`, `symbols.py`, `validation.py`. |
| State/docs | `document_store.py`, `context_retriever.py`, `wiki_indexer.py`, `settings.py`. |
| Backend services | `backend/services.py`. |
| Tests | Agent runtime, document store, RPC, ruff runner, services, widgets. |

Manifest RPC methods include document list/get/set/apply-edits plus `ruff_check`
and `ruff_fix`.

# Data And Provenance Impact

This plugin can mutate open documents and optionally run code-assistance or lint-fix
flows. It is not an MFDB plugin, but it is high-impact because file edits and agent
integration need a clear execution and safety model.

# Verification Surface

Focused test command:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/core/code_editor/test
```

# Documentation Work

- Add `chisurf/plugins/core/code_editor/README.md`.
- Document document-store semantics, edit application, and ruff fix behavior.
- State whether file writes happen immediately or through editor buffers.
- Document optional LSP/agent features separately from the core editor contract.
