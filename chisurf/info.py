import pathlib
import os
import re
import subprocess
from datetime import date
from typing import List, Optional


def _run_git(args: List[str], cwd: pathlib.Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    out = (out or "").strip()
    return out or None


def _normalize_tag_to_pep440(tag: str) -> Optional[str]:
    """Normalize a git tag like 'v26.0.0' to a PEP 440-compatible version.

    - Strips an optional leading 'v'
    - Removes leading zeros from numeric dot-separated segments
    - Returns None if the tag is not purely numeric/dot-separated
    """

    if not isinstance(tag, str):
        return None
    tag = tag.strip()
    if tag.startswith("v"):
        tag = tag[1:]
    if not tag:
        return None

    parts = tag.split(".")
    if not parts:
        return None
    normalized: List[str] = []
    for p in parts:
        if not p.isdigit():
            return None
        try:
            normalized.append(str(int(p)))
        except Exception:
            return None
    return ".".join(normalized)


def _compute_version() -> str:
    """Return a PEP 440-compatible version string.

    Version policy:
    - Prefer explicit override via CHISURF_VERSION.
    - If git tags are available, use `git describe`:
      - exact tag: '<tag>'
      - commits after tag: '<YY>.dev<COUNT>'
    - If no usable tags exist, fall back to '<YY>.dev<COUNT>'
      where COUNT is the commit count on HEAD (best-effort).
    """

    env_version = os.environ.get("CHISURF_VERSION")
    if env_version:
        return env_version.strip()

    repo_root = pathlib.Path(__file__).resolve().parent.parent

    desc = _run_git(["describe", "--tags", "--long", "--match", "v[0-9]*", "--dirty"], cwd=repo_root)
    if desc:
        m = re.match(r"^(v[0-9.]+)-(\d+)-g([0-9a-f]+)(-dirty)?$", desc)
        if m:
            base_tag = _normalize_tag_to_pep440(m.group(1))
            distance = int(m.group(2))
            dirty = bool(m.group(4))
            if base_tag:
                if distance == 0 and not dirty:
                    return base_tag
                # Dev snapshot after a tagged release; ensure it sorts *after*
                # the base release while still being a dev build.
                # Extract year from base tag for dev version
                base_parts = base_tag.split(".")
                if len(base_parts) >= 1:
                    year = base_parts[0]
                    return f"{year}.dev{distance}"
                else:
                    # Fallback to current year if tag format is unexpected
                    today = date.today()
                    year = str(int(today.strftime("%y")))
                    return f"{year}.dev{distance}"

    # Fallback: use current year and commit count
    today = date.today()
    year = str(int(today.strftime("%y")))

    count = _run_git(["rev-list", "--count", "HEAD"], cwd=repo_root)
    if not count or not count.isdigit():
        count = "0"

    return f"{year}.dev{count}"


today = date.today()

__name__ = "chisurf"
__author__ = "Thomas-Otavio Peulen"
__version__ = _compute_version()
__copyright__ = "Copyright (C) " + str(today.strftime('%y')) + " Thomas-Otavio Peulen"
__credits__ = ["Thomas-Otavio Peulen"]
__maintainer__ = "Thomas-Otavio Peulen"
__email__ = "thomas@peulen.xyz"
__url__ = "https://www.peulen.xyz/downloads/chisurf"
__license__ = 'GPL2.1'
__status__ = "Dev" if ("dev" in __version__) else "Release"
__description__ = """ChiSurf: an interactive global analysis platform for fluorescence data."""
__app_id__ = "{{ F25DCFFA-1234-4643-BC4F-2C3A20495937 }}"
LONG_DESCRIPTION = """ChiSurf: an interactive global analysis platform for fluorescence data."""
help_url = 'https://github.com/Fluorescence-Tools/chisurf/wiki'
update_url = 'https://github.com/Fluorescence-Tools/chisurf/releases'
setup_icon = "/gui/resources/icons/cs_logo.ico"
