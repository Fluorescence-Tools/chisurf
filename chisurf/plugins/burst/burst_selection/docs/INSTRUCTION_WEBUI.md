# Instruction: Electron/WebUI Planning & Prototype

## Goal

Replace the PyQt GUI with a local WebUI (Electron or similar) that
communicates with ChiSurf through the existing ZMQ JSON-RPC transport.

**Status:** This is a long-term objective. Do not start until:
- `INSTRUCTION_GUI_VERIFICATION.md` is complete (migrated PyQt passes desktop
  testing)
- `INSTRUCTION_TTTRLIB_BURSTFILTER.md` is complete (engine integration stable)
- All Burst Selection tests pass.

## Background

From `NEW_GUI_MIGRATION.md`:

> The future Electron/WebUI process should depend only on ChiSurf core and ZMQ
> JSON-RPC, not the PyQt GUI layer.

This means:

1. The WebUI is a **separate process** — it uses the ZMQ JSON-RPC client
   (`server/client.py`) without importing PyQt.
2. The same RPC methods (`burst_selection.analyze_files`,
   `burst_selection.inspect_bur`, `burst_selection.fit_gmm_from_bur`)
   are used by CLI, migrated PyQt GUI, and the WebUI.
3. Event topics (`burst_selection.job.*`) provide progress updates.

## Existing infrastructure

```bash
# ZMQ transport (already used by CLI `serve` command):
cat chisurf/server/transport/zmq.py

# Burst Selection client:
cat chisurf/plugins/burst/burst_selection/server/client.py

# ZMQ integration tests:
cat chisurf/plugins/burst/burst_selection/tests/test_server.py
```

## Planning tasks (do not implement until approved)

### 1. Technology selection

Evaluate:
- **Electron** — mature, heavy, good for desktop-native feel
- **Tauri** — Rust-based, lighter, good for smaller bundles
- **Python local web server** (FastAPI/Flask) + browser tab — simplest but
  no desktop integration

**Recommendation (from PRD):** Electron or similar framework.

### 2. Define WebUI scope

Minimum viable WebUI:

- TTTR file drag/drop or file browser
- Channel + filter settings (same as `gui/tool.py`)
- Run analysis → displays burst table
- Histogram with feature selection and GMM
- Save `.bur` output

### 3. Design: WebUI ↔ ChiSurf protocol

The WebUI calls:

```javascript
// JSON-RPC over ZMQ

// Analyze files
const result = await zmq.call("burst_selection.analyze_files", {
    files: ["/data/m000.spc"],
    settings: { photon_filter: { channels: [0,1,8,9] }, ... }
});

// Subscribe to progress
zmq.subscribe("burst_selection.job.progress", (msg) => {
    updateProgressBar(msg.progress, msg.message);
});

// Inspect a .bur file
const summary = await zmq.call("burst_selection.inspect_bur", {
    path: "/data/m000.bur"
});

// Fit GMM
const gmm = await zmq.call("burst_selection.fit_gmm_from_bur", {
    path: "/data/m000.bur",
    settings: { gmm: { n_components: 3 } }
});
```

### 4. Implementation phases

#### Phase A: Architecture decision

- Create a working prototype using a simple HTML page loaded from the ChiSurf
  server (served on an HTTP endpoint alongside ZMQ).
- Validate the JSON-RPC contract with real data.

#### Phase B: Electron shell

- Create an Electron app that opens the HTML UI.
- Electron spawns/connects to the ChiSurf ZMQ server.
- Implement file dialogs via Electron's native APIs.

#### Phase C: Feature parity

- Match all features of the migrated PyQt GUI.
- Add burst table sort/filter/search.
- Add plot interactivity (zoom, pan, export).

### 5. Integration with ChiSurf distribution

- The Electron app lives in its own repository or under
  `chisurf/plugins/burst/burst_selection/webui/`.
- ChiSurf's `pyproject.toml` may include an optional dependency or installer
  script for the WebUI bundle.

## Success criteria

- WebUI can analyze a TTTR file and display results identical to the migrated
  PyQt GUI.
- WebUI runs as a separate process (not embedded in ChiSurf).
- WebUI uses only `server/client.py` (ZMQ JSON-RPC) — no PyQt imports.
- Progress events flow from ChiSurf to WebUI.
- Electron build is distributable.

## Instruction status

This file is a **planning document only**. It becomes actionable when the
project lead approves the technology choice and scope.
