"""Compatibility wrapper for the ProteinMC model runner."""

from __future__ import annotations

from .model import (
    DirectLabelingPotential,
    ProteinMCProgress,
    ProteinMCResult,
    ProteinMCRunner,
    build_move_map_from_flexfit,
    list_flexfit_sets,
    load_json,
    load_structure,
    normalize_settings,
    run_protein_mc,
)


__all__ = [
    "DirectLabelingPotential",
    "ProteinMCProgress",
    "ProteinMCResult",
    "ProteinMCRunner",
    "build_move_map_from_flexfit",
    "list_flexfit_sets",
    "load_json",
    "load_structure",
    "normalize_settings",
    "run_protein_mc",
]


def main() -> None:
    """Run the ProteinMC command-line interface."""
    from .cli import cli

    cli.main(standalone_mode=True)


if __name__ == "__main__":
    main()
