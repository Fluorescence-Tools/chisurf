"""Command-line entry point for the FCS-Merger plugin."""

from __future__ import annotations

import argparse
import json

from ..core import merge_folder


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Merge / average FCS correlation curves.")
    p.add_argument("folder", help="Folder of .cor / .json.gz correlation chunks")
    p.add_argument("-o", "--output", default=None, help="Output .cor path")
    p.add_argument("--summary", action="store_true", help="Print only a short summary")
    args = p.parse_args(argv)

    result = merge_folder(args.folder, args.output)
    if args.summary:
        print(f"merged {result['n_curves']} curve(s), duration={result['duration']:g} s, "
              f"count_rate={result['count_rate']:g} kHz")
    else:
        print(json.dumps({k: v for k, v in result.items()
                          if k in ("duration", "count_rate", "n_curves")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
