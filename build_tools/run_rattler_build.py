#!/usr/bin/env python
"""Helper script to run rattler-build with consistent defaults."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path


def _slim_path() -> str:
    original = os.environ.get("PATH", "")
    parts = []

    def add(path: Path | str | None) -> None:
        if not path:
            return
        p = Path(path)
        if p.exists():
            parts.append(str(p))

    env_python = Path(sys.executable)
    env_root = env_python.parent
    add(env_root)
    add(env_root / "Scripts")
    add(env_root / "bin")
    add(env_root / "Library" / "bin")
    add(env_root / "Library" / "usr" / "bin")
    add(env_root / "Library" / "mingw-w64" / "bin")

    for tool in ("rattler-build", "git", "cmake"):
        tool_path = shutil.which(tool)
        if tool_path:
            add(Path(tool_path).parent)

    if platform.system() == "Windows":
        system_dirs = [
            Path("C:/Windows/system32"),
            Path("C:/Windows"),
            Path("C:/Windows/System32/Wbem"),
            Path("C:/Windows/System32/WindowsPowerShell/v1.0"),
            Path("C:/Windows/System32/OpenSSH"),
        ]
    else:
        system_dirs = [
            Path("/usr/local/sbin"),
            Path("/usr/local/bin"),
            Path("/usr/sbin"),
            Path("/usr/bin"),
            Path("/sbin"),
            Path("/bin"),
        ]
    for entry in system_dirs:
        add(entry)

    # Keep anything already short and critical from the existing PATH (like Visual Studio installers)
    for entry in original.split(os.pathsep):
        if entry and ("Microsoft Visual Studio" in entry or "Windows Kits" in entry):
            add(entry)

    deduped = OrderedDict((p, None) for p in parts if p)
    return os.pathsep.join(deduped.keys())


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    out_dir_env = os.environ.get("OUT_DIR")
    if out_dir_env:
        out_dir = Path(out_dir_env).expanduser().resolve()
    else:
        out_dir = (repo_root / "conda-bld").resolve()

    def ignore_patterns(path: str, names: list[str]) -> set[str]:
        ignored = set()
        for name in names:
            if name in {
                ".git",
                ".gitmodules",
                ".pixi",
                "conda-bld",
                "dist",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                "htmlcov",
            }:
                ignored.add(name)
        return ignored

    with tempfile.TemporaryDirectory(prefix="chisurf-rattler-src-") as tmp_dir:
        tmp_root = Path(tmp_dir) / "chisurf"
        shutil.copytree(repo_root, tmp_root, ignore=ignore_patterns)
        os.chdir(tmp_root)

        cmd = [
            "rattler-build",
            "build",
            "--recipe",
            str(tmp_root / "rattler-recipe"),
            "--output-dir",
            str(out_dir),
            "--test",
            "skip",
        ]

        env = os.environ.copy()
        env["PATH"] = _slim_path()

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False, env=env)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
