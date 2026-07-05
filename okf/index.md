---
okf_version: "0.1"
---

# ChiSurf Knowledge Bundle

An [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundle describing the ChiSurf codebase — its architecture, subsystems,
data stores, and developer workflows. Authored for agents and humans who
need durable context on how the repository is organized.

# Concepts

* [Overview](overview.md) - What ChiSurf is and how the source tree is laid out.

# Subdirectories

* [architecture](architecture/index.md) - The hybrid local/server design: API facade, action layer, runtime globals, plugin system, and the MFDB metadata store.
* [subsystems](subsystems/index.md) - The major code areas: core domain, GUI/AutoForm, headless server, and the plugin ecosystem.
* [plugins](plugins/index.md) - Plugin group pages, documentation standards, worklists, and high-priority plugin profiles.
* [workflows](workflows/index.md) - Developer workflows: environment/build with pixi, and running the test suites.
* [specs](specs/index.md) - Target ("north star") architecture specifications and the cleanup backlog tracking where today's code diverges from them.
* [prds](prds/index.md) - Product-requirement / design notes (PRD-NN) driving current work, integrated from `overhaul/`, each with persistent number and status.
* [references](references/index.md) - Pointers to maintained design docs and roadmap material.
