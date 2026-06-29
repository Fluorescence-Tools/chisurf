#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

from .export_fitting_parameters import _load_existing_registry, _save_registry


def _load_models(path: Path) -> Dict[str, Any]:
    """Load TCSPC parse-model definitions from tcspc.models.json.

    The JSON file is expected to map model names to dictionaries that contain
    at least an "initial" section with parameter names as keys.
    """

    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _collect_param_usage(models: Mapping[str, Any]) -> Dict[str, Set[str]]:
    """Return mapping of parameter name -> set of TCSPC model names using it."""

    usage: Dict[str, Set[str]] = {}
    for model_name, cfg in models.items():
        if not isinstance(cfg, dict):
            continue
        initial = cfg.get("initial", {})
        if not isinstance(initial, Mapping):
            continue
        for pname in initial.keys():
            if not isinstance(pname, str) or not pname:
                continue
            usage.setdefault(pname, set()).add(str(model_name))
    return usage


def _merge_tcspc_parameters(
    existing_params: Dict[str, Any],
    param_usage: Dict[str, Set[str]],
    origin_tag: str,
    family_prefix: str = "tcspc.",
) -> Dict[str, Any]:
    """Merge TCSPC parse-model parameters into an existing registry mapping.

    Each JSON parameter name ``p`` becomes a registry entry under
    ``f"{family_prefix}{p}"`` (e.g. ``tcspc.tau1``). Existing entries are
    preserved and only missing fields are filled. A single TCSPC-specific
    source record is added/updated per parameter to track the JSON origin and
    list of models that reference the parameter.
    """

    params = dict(existing_params) if isinstance(existing_params, dict) else {}

    for pname, models in sorted(param_usage.items()):
        reg_id = f"{family_prefix}{pname}"
        entry = params.get(reg_id)
        if not isinstance(entry, dict):
            entry = {}

        # Preserve existing description/keywords/aliases/label_texts/sources
        desc = entry.get("description")
        if not isinstance(desc, str):
            desc = ""

        keywords = entry.get("keywords")
        if not isinstance(keywords, list):
            keywords = []

        aliases = entry.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        if pname not in aliases:
            aliases.append(pname)

        label_texts = entry.get("label_texts")
        if not isinstance(label_texts, list):
            label_texts = []

        sources = entry.get("sources")
        if not isinstance(sources, list):
            sources = []

        # Merge/update a single TCSPC source record.
        existing_models: Set[str] = set()
        remaining_sources = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            fam = src.get("family")
            origin = src.get("origin")
            if fam == "tcspc" and origin == origin_tag:
                m = src.get("models")
                if isinstance(m, list):
                    for name in m:
                        if isinstance(name, str):
                            existing_models.add(name)
                # Drop this entry; it will be replaced with merged model list
                continue
            remaining_sources.append(src)

        all_models = sorted(existing_models.union(models))
        if all_models:
            remaining_sources.append(
                {
                    "family": "tcspc",
                    "origin": origin_tag,
                    "models": all_models,
                }
            )

        symbol = entry.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            symbol = pname

        updated = {
            "description": desc,
            "keywords": keywords,
            "symbol": symbol,
            "aliases": aliases,
            "label_texts": label_texts,
            "sources": remaining_sources,
        }
        params[reg_id] = updated

    return params


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate parameter_registry.json with skeleton entries for "
            "parse-based TCSPC decay models discovered in tcspc.models.json."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to the chisurf package root (defaults to this file's parent).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help=(
            "Path to the TCSPC models JSON file (defaults to core/models/tcspc/"
            "tcspc.models.json under --root)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path (defaults to chisurf/settings/constants/"
            "parameter_registry.json under --root)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    here = Path(__file__).resolve()
    default_root = here.parent.parent  # chisurf package directory

    source_root = Path(args.root).resolve() if args.root is not None else default_root

    if args.json is not None:
        json_path = Path(args.json).resolve()
    else:
        json_path = source_root / "core" / "models" / "tcspc" / "tcspc.models.json"

    if args.output is not None:
        out_path = Path(args.output).resolve()
    else:
        out_path = source_root / "settings" / "constants" / "parameter_registry.json"

    models = _load_models(json_path)
    existing = _load_existing_registry(out_path)

    if not models:
        _save_registry(out_path, existing)
        return

    param_usage = _collect_param_usage(models)

    try:
        origin_tag = str(json_path.relative_to(source_root.parent))
    except Exception:
        origin_tag = str(json_path)

    merged_params = _merge_tcspc_parameters(existing, param_usage, origin_tag=origin_tag)
    _save_registry(out_path, merged_params)


if __name__ == "__main__":
    main()
