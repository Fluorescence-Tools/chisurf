---
type: Architecture
title: MFDB — Metadata / Provenance Store
description: SQLite-backed metadata and provenance store with canonical tables generated from mmCIF dictionaries.
resource: chisurf/core/mfdb/
tags: [mfdb, metadata, provenance, sqlite, mmcif]
timestamp: '2026-07-05T00:00:00Z'
---

# Purpose

`chisurf/core/mfdb/` is a SQLite-backed metadata and provenance store. It is
versioned with migrations, has ACL/auth, and models a provenance DAG
(operations → artifacts → edges).

# Schema authority

The canonical tables are **generated from mmCIF dictionaries** (`data/*.dic` —
the wwPDB/PDB-IHM family plus the local `mfdb_flr_ext.dic`) by
`schema_from_dictionary.py`. Treat the `.dic` dictionaries as the schema
authority: change them and regenerate; do **not** hand-edit generated DDL.

# API

`api.py` is the transport-agnostic function API for the store.

# Related work

Curation of the fluorophore database (GUI/RPC/CLI) now lives in the
`core/mfdb_admin` plugin. MFDB is also the prototype for the broader
fdb4chembio (NFDI4Chem) fluorescence databank effort.

# Note on OKF

This [OKF](/index.md) bundle is a plain-markdown knowledge layer that sits
*beside* the code; MFDB is the runtime provenance store. OKF could serve as
an interchange/export format for MFDB provenance in the future.

# Citations

[1] [ChiSurf architecture doc](/references/architecture-doc.md)
