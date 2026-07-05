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
    if not isinstance(tag, str):
        return None
    tag = tag.strip()
    if tag.startswith("v"):
        tag = tag[1:]
    if not tag:
        return None

    # Handle v26.4.1-alpha.1 -> 26.4.1a1
    tag = tag.replace("-alpha.", "a").replace("-beta.", "b").replace("-rc.", "rc")
    tag = tag.replace("-alpha", "a").replace("-beta", "b").replace("-rc", "rc")

    pre_suffix = ""
    # Match suffixes like a1, b2, rc3
    m_pre = re.search(r"((?:a|b|rc)\d+)$", tag)
    if m_pre:
        pre_suffix = m_pre.group(1)
        tag = tag[:m_pre.start()]

    # Clean up any trailing dots or dashes before suffix
    tag = tag.rstrip(".-")

    parts = tag.split(".")
    if not parts:
        return None
    normalized: List[str] = []
    for p in parts:
        if p.isdigit():
            normalized.append(str(int(p)))
    
    if not normalized:
        return None
        
    return ".".join(normalized) + pre_suffix


def _find_git_root(start: pathlib.Path) -> Optional[pathlib.Path]:
    """Walk up from ``start`` looking for a ``.git`` entry (dir or file)."""
    for d in (start, *start.parents):
        if (d / ".git").exists():
            return d
    return None


def _git_version(repo_root: pathlib.Path) -> Optional[str]:
    """Derive a PEP 440 version from git history, or None if unavailable."""
    desc = _run_git(["describe", "--tags", "--long", "--match", "v[0-9]*", "--dirty"], cwd=repo_root)
    if desc:
        dirty = desc.endswith("-dirty")
        clean_desc = desc[:-6] if dirty else desc

        # Split from the right to separate distance and hash
        parts = clean_desc.split("-")
        if len(parts) >= 3:
            # Format: <tag>-<distance>-g<hash>
            # But <tag> could also contain dashes!
            distance_str = parts[-2]
            tag_part = "-".join(parts[:-2])

            if distance_str.isdigit():
                distance = int(distance_str)
                base_tag = _normalize_tag_to_pep440(tag_part)

                if base_tag:
                    if distance == 0 and not dirty:
                        return base_tag
                    # Dev snapshot after a tagged release
                    base_parts = base_tag.split(".")
                    if len(base_parts) >= 1:
                        year = base_parts[0]
                        return f"{year}.dev{distance}"

    # No reachable tag (e.g. branch diverged from tagged history): fall back to
    # the current year and the commit count on HEAD.
    count = _run_git(["rev-list", "--count", "HEAD"], cwd=repo_root)
    if count and count.isdigit():
        year = str(int(date.today().strftime("%y")))
        return f"{year}.dev{count}"

    return None


def _compute_version() -> str:
    """Return a PEP 440-compatible version string.

    Version policy:
    - Prefer explicit override via CHISURF_VERSION.
    - In a dev checkout (a ``.git`` directory is present), always derive from git
      so a stale build-time ``_version.py`` artifact can never win here:
      - exact tag: '<tag>'
      - commits after tag: '<YY>.dev<COUNT>'
      - no reachable tag: '<YY>.dev<COUNT>' where COUNT is the HEAD commit count.
    - Otherwise (installed app, no git) use the build-time static
      ``chisurf/core/_version.py`` if present: this avoids spawning git on every
      ``import chisurf``.
    - As a last resort, fall back to '<YY>.dev0'.
    """

    env_version = os.environ.get("CHISURF_VERSION")
    if env_version:
        return env_version.strip()

    # Dev checkout: git is authoritative. Checking for ``.git`` first keeps the
    # fast path intact for installed apps (no git subprocess when there is no
    # repo to read).
    package_dir = pathlib.Path(__file__).resolve().parent.parent
    git_root = _find_git_root(package_dir)
    if git_root is not None:
        version = _git_version(git_root)
        if version:
            return version

    # Build-time static version (written by build_installer.py). Fast path for
    # installed apps: no git subprocess at import time.
    try:
        from chisurf.core._version import __version__ as _static_version
        if _static_version:
            return _static_version
    except Exception:
        pass

    # Last resort.
    year = str(int(date.today().strftime("%y")))
    return f"{year}.dev0"


today = date.today()

__name__ = "chisurf"
__author__ = "Thomas-Otavio Peulen"
__version__ = _compute_version()
__copyright__ = "Copyright (C) " + str(today.strftime('%y')) + " Thomas-Otavio Peulen"
__credits__ = ["Thomas-Otavio Peulen"]
__maintainer__ = "Thomas-Otavio Peulen"
__email__ = "thomas@peulen.xyz"
__url__ = "https://www.peulen.xyz/downloads/chisurf"
__license__ = 'GPL2.0'
__status__ = "Dev" if ("dev" in __version__) else "Release"
__description__ = """ChiSurf: an interactive global analysis platform for fluorescence data."""
__app_id__ = "{{ F25DCFFA-1234-4643-BC4F-2C3A20495937 }}"
LONG_DESCRIPTION = """ChiSurf: an interactive global analysis platform for fluorescence data."""
help_url = 'https://github.com/Fluorescence-Tools/chisurf/wiki'
update_url = 'https://github.com/Fluorescence-Tools/chisurf/releases'
setup_icon = "/gui/resources/icons/cs_logo.ico"
