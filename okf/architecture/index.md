# Architecture

* [API Facade](api-facade.md) - `ChiSurfAPI`, the stable local/hybrid/server facade for GUI, macros, plugins, and the console.
* [Action Layer](action-layer.md) - `ActionRegistry`/`ActionDispatcher` mediating all state changes.
* [Runtime Globals](runtime-globals.md) - Legacy process-local globals in `chisurf/__init__.py` and the migration away from them.
* [Server](server.md) - The headless, Qt-free ZMQ/JSON-RPC server under `chisurf/server/`.
* [Plugin System](plugin-system.md) - Manifest-discovered plugins and the plugin infrastructure.
* [MFDB Metadata Store](mfdb.md) - SQLite-backed metadata/provenance store generated from mmCIF dictionaries.
