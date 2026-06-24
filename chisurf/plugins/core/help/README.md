# ChiSurf Help Plugin

Documentation browser and help resource viewer for ChiSurf.

## Architecture

This plugin follows the **new standard architecture** with clean separation:

```
help/
  manifest.json         Plugin manifest (source of truth)
  __init__.py           Package init, loads manifest
  api/                  Pure Python layer (no Qt)
    contract.py         Workflow contract & JSON schemas
    models.py           Dataclass models
    markdown.py         Markdown rendering (with/without `markdown` package)
    io.py               Document discovery, reading, saving, search
  backend/              ZMQ RPC handlers
    services.py         RPC method registration
    state.py            Plugin state dataclass
  gui/                  Qt GUI layer
    tool.py             HelpWidget (QMainWindow with toolbar)
    client.py           HelpClient wrapping PluginClient
  cli/                  Click-based CLI
    main.py             CLI commands (list, read, render, search)
  test/                 Pytest tests
    test_widgets.py     HelpWidget creation test
```

## Features

- **Tree navigation** of user manual, core docs, and per-plugin documentation
- **Full-text search** across all documents with real-time filter
- **In-app Markdown editing** with toggle between View/Edit
- **Toolbar buttons** with emoji icons:
  - ✏️ Edit / 👁️ View toggle
  - 💾 Save document edits
  - 📖 Open Docs — online documentation in browser
  - 🎬 Video Tutorials — tutorial videos in browser
  - ❌ Close — dismiss help window
- **Context-sensitive help** from experiments and settings panels

## CLI Usage

```bash
python -m chisurf help list         # List all docs
python -m chisurf help read <path>  # Read a doc file
python -m chisurf help render <path> # Render doc to HTML
python -m chisurf help search <q>   # Search docs
```

## RPC Methods

| Method | Description |
|--------|-------------|
| `help.docs.list` | List all documentation files |
| `help.docs.read` | Read and render a document |
| `help.docs.save` | Save edited document content |
| `help.docs.search` | Search across documents |
| `help.docs.contract` | Return workflow contract |

## License

Part of the ChiSurf package. Distributed under the same license.
