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
        dirty = desc.endswith("-dirty")
        clean_desc = desc[:-6] if dirty else desc
        
        # Split from the right to separate distance and hash
        parts = clean_desc.split("-")
        if len(parts) >= 3:
            # Format: <tag>-<distance>-g<hash>
            # But <tag> could also contain dashes!
            git_hash = parts[-1]
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
__license__ = 'GPL2.0'
__status__ = "Dev" if ("dev" in __version__) else "Release"
__description__ = """ChiSurf: an interactive global analysis platform for fluorescence data."""
__app_id__ = "{{ F25DCFFA-1234-4643-BC4F-2C3A20495937 }}"
LONG_DESCRIPTION = """ChiSurf: an interactive global analysis platform for fluorescence data."""
help_url = 'https://github.com/Fluorescence-Tools/chisurf/wiki'
update_url = 'https://github.com/Fluorescence-Tools/chisurf/releases'
setup_icon = "/gui/resources/icons/cs_logo.ico"
