# ChiSurf Assistant Chat Plugin

Local, CPU‑only in‑app assistant for ChiSurf. Ships as a PyQt Dock widget that talks to a local FastAPI backend to:
- Answer questions using Retrieval‑Augmented Generation (RAG) over ChiSurf docs and code
- Execute registered tools (e.g., anisotropy analysis) via deterministic JSON tool‑calls
- Optionally run in a bounded Agent mode (plan → act → observe; max 4 steps)

This README covers how to use the plugin and how to build the RAG context index.


## Quick start (ChiSurf UI)
- Open ChiSurf.
- From the Plugins menu, launch: Help → Assistant Chat.
- A dock named “ChiSurf Assistant” appears on the right.
- Type a question or instruction into the input line and click Send.
  - For single‑turn answers/actions, leave “Agent Mode” unchecked.
  - For multi‑step workflows (list → confirm → batch), enable “Agent Mode”.

The dock will automatically start the local backend if it’s not running.


## Building the RAG context (indexing)
The assistant answers strictly from your local context. Build the RAG index once (and rebuild when your docs/code change):

- From a terminal at the project root (E:\dev\chisurf):

  ```powershell
  python -m chisurf.plugins.chat.index_docs --store rag_store
  ```

What gets indexed:
- Markdown: docs/**/*.md, README*.md
- Word: docs/**/*.docx (paragraph text)
- Source code: src/**/*.{py,cpp,h,hpp}, chisurf/**/*, modules/**/*
  - Extracts docstrings ("""..."""/'''...''') and line comments (# ...)
  - Falls back to entire file text if none found

Chunking:
- Chunk size ≈ 900 characters with ≈ 200 overlap

Embeddings and index:
- If Ollama is available locally, uses nomic-embed-text (dim=768)
- Otherwise, uses a deterministic, CPU‑only pseudo‑embedding (no network)
- FAISS IndexFlatIP is used when faiss-cpu is installed; otherwise a NumPy fallback is used
- The store is persisted to rag_store/ with meta.json, texts.jsonl, embeddings.npy, and optionally index.faiss

Retrieval formatting (as seen in /chat responses):

```
[<absolute-path-1>]
<chunk-1>

---

[<absolute-path-2>]
<chunk-2>
...
```


## Backend service (FastAPI)
The dock tries to autostart the backend if it cannot connect. You can also start it manually:

```powershell
python -m chisurf.plugins.chat.server
# http://127.0.0.1:8000/
```

Endpoints:
- POST /chat → RAG + single response or tool execution
- POST /agent → bounded agent loop (≤ 4 steps)
- POST /analyze-file → direct anisotropy run (upload CSV or provide csv_path via form fields)

System prompt guardrails enforced by the server:
- Answer strictly using provided context
- Never invent file paths; list → select → confirm
- For actions, emit tool calls only using the fenced JSON format shown below
- Request explicit confirmation before destructive actions


## Deterministic tool‑call format
The assistant must emit tool calls using an exact fenced block:

````
```toolcall
{
  "tool": "run_anisotropy_analysis",
  "args": {
    "csv_path": "C:\\Data\\exp42\\trace_001.csv",
    "g_factor": 0.98,
    "time_col": "time",
    "Ipar_col": "I_par",
    "Iperp_col": "I_perp",
    "smooth_window": 5,
    "out_prefix": "anisotropy_out"
  }
}
```
````

The backend parses this block, validates arguments against the tool’s JSON schema, enforces path whitelisting, and executes safely.


## Tools available
- run_anisotropy_analysis (non‑destructive):
  - Args: csv_path (absolute), g_factor, time_col, Ipar_col, Iperp_col, smooth_window, out_prefix
  - Computes r(t) = (I_par − G·I_perp) / (I_par + 2·G·I_perp)
  - Avoids division by zero; optional moving‑average smoothing
  - Saves {out_prefix}.csv and {out_prefix}.png next to the input by default
  - Returns summary (n_points, r_mean, r_std, r_min, r_max, output paths)

- list_files (non‑destructive):
  - Args: dir_path (absolute), pattern="*.csv", max_items (≥1)
  - Returns absolute file paths up to max_items

- read_text_head (non‑destructive):
  - Args: path (absolute), n (default 2000)
  - Returns the first n characters of a UTF‑8 text file

- batch_run_anisotropy (destructive):
  - Args: dir_path (absolute), pattern, and same analysis options as single‑file
  - Runs anisotropy on each matched file; saves outputs alongside inputs
  - Returns a table per file (r_mean, r_std, output paths)
  - Requires confirmation (confirm_destructive=true)


## Agent mode (optional)
- Enable “Agent Mode” in the dock for short multi‑step flows.
- The agent is bounded to 4 steps and at most one tool call per turn.
- Typical flow for batch processing:
  1) list_files → 2) ask to confirm → 3) batch_run_anisotropy (requires confirmation) → 4) summarize
- If the backend responds that confirmation is required, the dock shows “Confirm Action”. Clicking it resends with confirm_destructive=true.

Example goal:
```
Goal: Compute anisotropy for all CSVs in D:\data\exp42 with G=0.98, save outputs next to inputs, then summarize r_mean and r_std per file.
Constraints: read‑only until I confirm. Never invent paths; list first.
Success: a compact table plus output file paths.
```


## Safety, whitelisting, and logging
- Path safety: Arguments like csv_path, dir_path, path must be absolute and within whitelisted roots.
  - Default whitelist includes the project root and test/data.
  - You can extend it per request by passing a whitelist array in the JSON body.
- Destructive tools (e.g., batch_run_anisotropy) are blocked unless confirm_destructive=true is provided.
- Backend logs: request type, retrieved paths, chosen tool, args, duration, outcome (no secrets).
- UI logs: user message, assistant reply, tool results (summarized).


## Dependencies (runtime)
These should already be part of ChiSurf’s environment via the conda recipe:
- fastapi, uvicorn, httpx, python-multipart
- faiss-cpu (optional but recommended; NumPy fallback is supported)
- python-docx, numpy, pandas, matplotlib, PyQt5

CPU‑only, no external network calls. Ollama is optional and local only (127.0.0.1:11434).

## Install Ollama (required for chat)
- Windows:
  - Download and install from https://ollama.com/download or use winget: `winget install Ollama.Ollama`
- macOS:
  - Homebrew: `brew install ollama`
- Linux:
  - Follow the instructions at https://ollama.com/download (one-line install script) or use your distro packages if available.

Start the local server:
```
ollama serve
```

Pull the required models:
```
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Then try the Assistant Chat again. The backend connects to Ollama at 127.0.0.1:11434 by default.

If you want to use local LLMs via Ollama:
- Chat model: llama3.2:1b
- Embeddings: nomic-embed-text
- Pull models (optional):
  ```
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
  ```

The indexer and backend automatically fall back to deterministic embeddings and heuristic replies if Ollama isn’t available.


## Troubleshooting
- “Connection refused” from UI: the dock should autostart the backend; try again after a second. If it persists, start manually:
  ```
  python -m chisurf.plugins.chat.server
  ```
- Empty answers or no citations: ensure you built the index:
  ```
  python -m chisurf.plugins.chat.index_docs --store rag_store
  ```
- Anisotropy errors: check CSV has the expected columns (default: time, I_par, I_perp). Adjust column names via tool args.
- Whitelist rejections: pass additional allowed roots in the request body (whitelist=["C:\\Data\\...\"]).


## Developer notes
- Dock widget: chisurf/plugins/chat/qt_chat_dock.py (class ChiSurfChatDock)
- Backend: chisurf/plugins/chat/server.py (APP)
- Indexer: chisurf/plugins/chat/index_docs.py
- Tools: chisurf/plugins/chat/tools/anisotropy.py
- Agent loop/registry: chisurf/plugins/chat/agent/

Unit tests (subset) live under test/test_assistant_plugin.py and exercise indexing, tool schema validation, anisotropy outputs, and agent confirmation flow.