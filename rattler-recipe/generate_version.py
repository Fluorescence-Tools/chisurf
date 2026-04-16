import os
import sys
import json
import pathlib
import subprocess
import re
from datetime import date


def _run_git(args, cwd):
    try:
        out = subprocess.check_output(
            ["git"] + args,
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    out = (out or "").strip()
    return out or None


def _normalize_tag_to_pep440(tag):
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
    normalized = []
    for p in parts:
        if p.isdigit():
            normalized.append(str(int(p)))
    
    if not normalized:
        return None
        
    return ".".join(normalized) + pre_suffix


def _compute_version():
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
                    base_parts = base_tag.split(".")
                    if len(base_parts) >= 1:
                        year = base_parts[0]
                        return f"{year}.dev{distance}"

    today = date.today()
    year = str(int(today.strftime("%y")))
    count = _run_git(["rev-list", "--count", "HEAD"], cwd=repo_root)
    if not count or not count.isdigit():
        count = "0"
    return f"{year}.dev{count}"


version = _compute_version()

if "--print" in sys.argv:
    print(version)
