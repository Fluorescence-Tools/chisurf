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

    pre_suffix = ""
    m_pre = re.match(r"^(.+?)((?:a|b|rc)\d+)$", tag)
    if m_pre:
        tag = m_pre.group(1)
        pre_suffix = m_pre.group(2)

    parts = tag.split(".")
    if not parts:
        return None
    normalized = []
    for p in parts:
        if not p.isdigit():
            return None
        try:
            normalized.append(str(int(p)))
        except Exception:
            return None
    return ".".join(normalized) + pre_suffix


def _compute_version():
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    desc = _run_git(["describe", "--tags", "--long", "--match", "v[0-9]*", "--dirty"], cwd=repo_root)
    if desc:
        m = re.match(r"^(v[\d.]+(?:a|b|rc)?\d*)-(\d+)-g([0-9a-f]+)(-dirty)?$", desc)
        if m:
            base_tag = _normalize_tag_to_pep440(m.group(1))
            distance = int(m.group(2))
            dirty = bool(m.group(4))
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

version_file = pathlib.Path(__file__).resolve().parent / "version.json"
if "--print" in sys.argv:
    print(version)
else:
    with open(version_file, "w") as f:
        json.dump({"version": version}, f, indent=2)
    print(f"Generated {version_file} with version: {version}")
