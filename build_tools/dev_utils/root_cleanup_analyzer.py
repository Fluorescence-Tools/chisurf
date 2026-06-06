"""
Root Cleanup Analyzer
---------------------
Scans the project root for top-level files and directories and classifies them
as likely keepers or candidates for archival/removal based on simple heuristics
and whether they are referenced anywhere in the repository.

Output: ROOT_CLEANUP_REPORT.md in the repository root.

This script is intentionally conservative — it does not delete anything.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Dict, List, Tuple, Set


ROOT = Path(__file__).resolve().parents[1]


# Files and directories that are almost always keepers in a Python project
ALWAYS_KEEP_FILES: Set[str] = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "MANIFEST.in",
    "opencode.json",
}

ALWAYS_KEEP_DIRS: Set[str] = {
    # Core code
    "chisurf",
    "modules",
    # Documentation and build tooling
    "docs",
    "build_tools",
    # Tests and notebooks
    "test",
    "unittests",
    "notebooks",
    # Project metadata
    "manual",
}


# Globs or exact names that are often byproducts and can be archived/cleaned
LIKELY_NOISE_PATTERNS = [
    re.compile(r"^benchmark_.*\\.py$", re.I),
    re.compile(r"^benchmark_.*\\.json$", re.I),
    re.compile(r"^run_output_.*\\.log$", re.I),
    re.compile(r"^pytest_.*\\.(log|txt|json)$", re.I),
    re.compile(r"^test_out\\.txt$", re.I),
    re.compile(r"^.*\\.(ipynb_checkpoints|tmp)$", re.I),
]


IGNORE_DIRS = {
    ".git",
    ".idea",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
}


TEXT_FILE_EXTS = {
    ".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".gitignore", ".bat", ".ps1",
}


def is_text_file(path: Path) -> bool:
    if path.is_dir():
        return False
    if path.suffix.lower() in TEXT_FILE_EXTS:
        return True
    # Heuristic fallback: try reading small chunk
    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
        chunk.decode("utf-8")
        return True
    except Exception:
        return False


def iter_project_files(base: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(base):
        # prunes
        rel_parts = Path(root).relative_to(base).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            # Skip whole subtree by mutating dirs in-place
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for name in files:
            p = Path(root) / name
            yield p


def ref_search_terms(name: str) -> List[str]:
    stem = Path(name).stem
    terms = {name, stem}
    # Add variants without spaces/underscores
    compact = re.sub(r"[\s_\-]+", "", stem)
    if compact and compact != stem:
        terms.add(compact)
    return list(terms)


def find_references(root_items: List[Path]) -> Dict[Path, List[Tuple[Path, str]]]:
    refs: Dict[Path, List[Tuple[Path, str]]] = defaultdict(list)
    # Build quick text index: scan each file once and check all names
    items_by_term: Dict[str, List[Path]] = defaultdict(list)
    for item in root_items:
        for term in ref_search_terms(item.name):
            items_by_term[term.lower()].append(item)

    for p in iter_project_files(ROOT):
        if p == ROOT / "ROOT_CLEANUP_REPORT.md":
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".ptu"}:
            continue
        if not is_text_file(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lower = text.lower()
        for term, items in items_by_term.items():
            if term and term in lower:
                for item in items:
                    refs[item].append((p, term))
    return refs


def classify_item(p: Path, has_refs: bool) -> str:
    name = p.name
    if p.is_dir():
        if name in ALWAYS_KEEP_DIRS:
            return "KEEP"
        # Non-core directories at root: likely project-support or legacy
        return "REVIEW"
    # Files
    if name in ALWAYS_KEEP_FILES:
        return "KEEP"
    for pat in LIKELY_NOISE_PATTERNS:
        if pat.match(name):
            return "ARCHIVE"
    if name.lower().endswith((".ipynb",)):
        return "ARCHIVE"
    if name.lower().endswith((".log", ".csv")):
        return "ARCHIVE"
    if name.lower().endswith((".md",)):
        # Design docs at root are ok — keep if referenced, else review
        return "KEEP" if has_refs else "REVIEW"
    if name.lower().endswith((".py",)):
        # Standalone scripts at root: keep if referenced, else review/archive
        return "KEEP" if has_refs else "REVIEW"
    # Default
    return "KEEP" if has_refs else "REVIEW"


def main(argv: List[str]) -> int:
    root_items = [p for p in ROOT.iterdir() if p.name not in {".git", ".idea", "venv", ".venv"}]
    # Only consider first-level items in root
    refs = find_references(root_items)

    lines: List[str] = []
    lines.append("### Root Cleanup Report")
    lines.append("")
    lines.append(f"Generated by dev_tools/root_cleanup_analyzer.py")
    lines.append("")
    lines.append("Legend: KEEP = essential/used, REVIEW = needs human decision, ARCHIVE = safe to move to archive/, REMOVE = safe to delete")
    lines.append("")

    buckets: Dict[str, List[Path]] = defaultdict(list)
    details: Dict[Path, Tuple[str, List[Tuple[Path, str]]]] = {}
    for item in sorted(root_items, key=lambda p: (0 if p.is_dir() else 1, p.name.lower())):
        has_refs = len(refs.get(item, [])) > 0
        cls = classify_item(item, has_refs)
        buckets[cls].append(item)
        details[item] = (cls, refs.get(item, []))

    order = ["KEEP", "REVIEW", "ARCHIVE"]
    for cls in order:
        items = buckets.get(cls, [])
        if not items:
            continue
        lines.append(f"#### {cls}")
        lines.append("")
        for item in items:
            dcls, rlist = details[item]
            ref_info = " (refs: {} files)".format(len(rlist)) if rlist else ""
            kind = "DIR" if item.is_dir() else "FILE"
            lines.append(f"- {kind} {item.name}{ref_info}")
            # Show up to 5 reference paths
            if rlist:
                for ref_path, term in rlist[:5]:
                    rel = ref_path.relative_to(ROOT)
                    lines.append(f"  - used in {rel} (term '{term}')")
        lines.append("")

    # Suggestions
    lines.append("### Suggested Actions")
    lines.append("")
    lines.append("- Move ARCHIVE items into an `archive/` folder in the repo root to keep history while decluttering.")
    lines.append("- For REVIEW items, decide whether they belong under docs/, notebooks/, or dev_tools/, or can be removed.")
    lines.append("- Keep KEEP items as they are.")
    lines.append("")

    out = ROOT / "ROOT_CLEANUP_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
