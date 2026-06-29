"""Shared scaffolding so every ``download/*`` scraper is structured the same way.

Two things live here:

- :data:`SCRAPERS` — the single registry of source scrapers (label, canonical
  source slug, whether ``run-all`` includes it by default). The CLI and the
  download-manager GUI both discover sources from this registry instead of
  globbing the directory or hardcoding lists.
- :func:`scraper_main` — the uniform CLI entry point each scraper module calls
  from ``__main__``: it parses ``--db`` (plus any scraper-specific options),
  opens the staging :class:`FluorophoreDatabase`, and runs the scraper inside a
  ``with db:`` block. Scrapers therefore share one consistent shape.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)


@dataclass(frozen=True)
class ScraperSpec:
    """Declarative description of one source scraper (wiring only)."""

    module: str       # download/<module>.py (run via `python -m ...download.<module>`)
    label: str        # human-readable name
    source: str       # canonical provenance slug stamped on its probes
    default: bool = True  # included in `run-all` when no --only subset is given


# The registry — add a scraper here and the CLI + GUI pick it up. ``default`` is
# False for sources that are currently non-functional (so `run-all` skips them
# unless explicitly requested via --only).
SCRAPERS: list[ScraperSpec] = [
    ScraperSpec("fpbase", "FPbase", "fpbase"),
    ScraperSpec("chroma", "Chroma", "chroma"),
    ScraperSpec("thorlabs", "Thorlabs", "thorlabs"),
    ScraperSpec("photochemcad_common_compounds", "PhotochemCAD", "photochemcad"),
    # ATTO-TEC's live site (now Leica) dropped the per-dye spectra; this scraper
    # recovers them from the Internet Archive (Wayback Machine).
    ScraperSpec("atto", "ATTO-TEC (Wayback)", "atto"),
    # 3DOptix carries no spectra — it is a metadata/optical-property enricher
    # (cut-on/cut-off, material, shape …) that enriches matching components via
    # dedup. Off by default: it is a slow crawl run deliberately, not a spectrum
    # source.
    ScraperSpec("threed_optix", "3DOptix (metadata only)", "3doptix", default=False),
    # Omega's example product URLs are dead. Kept in the registry but off by default.
    ScraperSpec("omega_optical", "Omega Optical", "omega", default=False),
]

_BY_MODULE = {s.module: s for s in SCRAPERS}


def get_scraper(module: str) -> ScraperSpec | None:
    """Return the :class:`ScraperSpec` for a module name, or ``None``."""
    return _BY_MODULE.get(module)


def default_scrapers() -> list[ScraperSpec]:
    """Scrapers included in ``run-all`` when no explicit subset is given."""
    return [s for s in SCRAPERS if s.default]


def scraper_main(
    description: str,
    run: Callable[[FluorophoreDatabase, argparse.Namespace], object],
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Uniform ``__main__`` for a scraper module.

    Parameters
    ----------
    description
        ``argparse`` description.
    run
        ``run(db, args)`` — called inside a ``with db:`` block; performs the
        download+sort and writes to the staging DB via ``register_component``.
    add_arguments
        Optional hook to register scraper-specific CLI options on the parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--db", default=str(DEFAULT_DATABASE_PATH),
        help="Staging spectra.db path (default: the bundled reference DB).",
    )
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()

    db = FluorophoreDatabase(args.db)
    with db:
        run(db, args)
