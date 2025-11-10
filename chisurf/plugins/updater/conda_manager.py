import os
import sys
import json
import subprocess
import platform
import re
from typing import List, Dict, Any, Optional, Tuple

# Reuse conda discovery logic from the updater
from .updater import ChiSurfUpdater


class CondaCommandError(Exception):
    def __init__(self, message: str, stdout: str = "", stderr: str = "", returncode: int = -1):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class CondaManager:
    """
    Minimal Conda package/environment manager for ChiSurf.

    This class wraps the conda CLI (preferring JSON output) to provide basic
    operations needed by the ChiSurf updater UI:
      - Query conda info and environments
      - List/search packages
      - Install/update/remove packages in a selected environment
      - (Optionally) create/remove environments (not wired into UI initially)

    It intentionally avoids external heavy dependencies and relies on the
    conda executable available in the current installation.
    """

    def __init__(self, conda_executable: Optional[str] = None):
        self._system = platform.system().lower()
        if conda_executable:
            self.conda_exe = conda_executable
        else:
            # Leverage existing discovery logic
            self.conda_exe = ChiSurfUpdater(update_url=None)._get_conda_executable()

    # --- Internal helpers -------------------------------------------------
    def _wrap_cmd_for_windows(self, cmd: List[str]) -> List[str]:
        if self._system == 'windows':
            exe = (cmd[0] if cmd else '').lower()
            if exe.endswith('.bat') or exe.endswith('.cmd'):
                return ['cmd.exe', '/C', *cmd]
        return cmd

    def _truncate(self, text: str, limit: int = 4000) -> str:
        if text and len(text) > limit:
            return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
        return text or ""

    def _sanitize_json_text(self, text: str) -> str:
        """
        Try to extract a valid JSON document from polluted stdout.
        - Strips BOM/whitespace
        - Drops any leading text before the first '{' or '['
        - Trims trailing junk after the last matching '}' or ']'
        """
        if not isinstance(text, str):
            return ""
        s = text.lstrip("\ufeff\r\n\t ")
        # Find first plausible JSON start
        lb = s.find('[')
        lb = lb if lb != -1 else 10**9
        lb2 = s.find('{')
        lb2 = lb2 if lb2 != -1 else 10**9
        start = min(lb, lb2)
        if start == 10**9:
            return s
        s = s[start:]
        # Find last plausible JSON end
        rb = s.rfind(']')
        rb2 = s.rfind('}')
        end = max(rb, rb2)
        if end != -1:
            s = s[:end+1]
        return s

    def _parse_json_relaxed(self, text: str) -> Any:
        """
        Attempt to parse JSON from text using relaxed strategies:
        - Direct json.loads
        - After sanitization of leading/trailing junk
        - NDJSON-style: parse lines that look like JSON objects and return a list
        - Concatenated objects: split on '\n}\n{\n' or '}\n{' and wrap into list
        """
        if not text:
            return {}
        # 1) Direct
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2) Sanitize leading/trailing
        s = self._sanitize_json_text(text)
        try:
            return json.loads(s)
        except Exception:
            pass
        # 3) Concatenated objects -> list
        # Try to detect occurrences of '}{' or pattern with newlines
        if ('}{' in s) or ('}\n{' in s):
            parts = re.split(r'}\s*\n?\s*{', s)
            objs: List[Any] = []
            for i, p in enumerate(parts):
                # Re-add braces lost by split
                if not p.strip():
                    continue
                frag = ('{' + p if i > 0 else p)
                if not frag.strip().startswith('{'):
                    frag = '{' + frag
                if not frag.strip().endswith('}'):
                    frag = frag + '}'
                try:
                    obj = json.loads(frag)
                    objs.append(obj)
                except Exception:
                    continue
            if objs:
                return objs
        # 4) NDJSON: try line by line objects
        lines = [ln.strip() for ln in s.splitlines() if ln.strip().startswith('{') and ln.strip().endswith('}')]
        items: List[Any] = []
        for ln in lines:
            try:
                items.append(json.loads(ln))
            except Exception:
                continue
        if items:
            return items
        # Give up
        raise json.JSONDecodeError("Unable to parse JSON (relaxed)", s, 0)

    def _run(self, args: List[str], use_json: bool = True, env: Optional[Dict[str, str]] = None):
        cmd = [self.conda_exe, *args]
        # Prefer quiet mode to reduce non-JSON noise
        if '--quiet' not in cmd and '-q' not in cmd:
            cmd.append('--quiet')
        if use_json and '--json' not in cmd:
            cmd.append('--json')
        popen_cmd = self._wrap_cmd_for_windows(cmd)
        try:
            proc = subprocess.Popen(
                popen_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                shell=False,
                env=env or os.environ.copy(),
            )
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                # Try to parse JSON error if present (relaxed)
                msg = None
                if stdout:
                    try:
                        data = self._parse_json_relaxed(stdout)
                        if isinstance(data, dict):
                            msg = data.get('error') or data.get('message') or data.get('exception_name')
                    except Exception:
                        pass
                if not msg:
                    msg = (stderr or '').strip() or 'Conda command failed'
                raise CondaCommandError(msg, self._truncate(stdout), self._truncate(stderr), proc.returncode)
            if use_json:
                # Strict first
                try:
                    return json.loads(stdout or '{}')
                except Exception:
                    # Relaxed parsing
                    try:
                        return self._parse_json_relaxed(stdout)
                    except Exception as e:
                        raise CondaCommandError(f"Invalid JSON from conda: {e}\nOutput: {self._truncate(stdout)}",
                                                self._truncate(stdout), self._truncate(stderr), proc.returncode)
            else:
                # Return plain text wrapped in a dict
                return {"stdout": stdout, "stderr": stderr}
        except CondaCommandError:
            raise
        except Exception as e:
            raise CondaCommandError(f"Failed to run conda: {e}")

    def _prefix_args(self, name: Optional[str] = None, prefix: Optional[str] = None) -> List[str]:
        args: List[str] = []
        if prefix:
            args += ['--prefix', prefix]
        elif name:
            args += ['--name', name]
        return args

    # --- Public API -------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return self._run(['info'])

    def get_envs(self) -> Dict[str, Any]:
        """
        Returns a dict with keys:
          - envs: List[str]
          - default_prefix: str
          - active_prefix: str (when available)
        """
        data = self._run(['info'])
        envs = data.get('envs', [])
        default_prefix = data.get('default_prefix') or ''
        active_prefix = os.environ.get('CONDA_PREFIX', '') or data.get('active_prefix', '')
        return {
            'envs': envs,
            'default_prefix': default_prefix,
            'active_prefix': active_prefix,
        }

    def _parse_list_plaintext(self, text: str) -> List[Dict[str, Any]]:
        """Parse plain text output from `conda list` into records.
        Expected columns: name, version, build, [channel]. Ignores commented lines.
        """
        records: List[Dict[str, Any]] = []
        if not text:
            return records
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 3:
                rec: Dict[str, Any] = {
                    'name': parts[0],
                    'version': parts[1],
                    'build_string': parts[2],
                }
                if len(parts) >= 4:
                    rec['channel'] = parts[3]
                records.append(rec)
        return records

    def list_packages(self, prefix: Optional[str] = None, name: Optional[str] = None) -> List[Dict[str, Any]]:
        args = ['list'] + self._prefix_args(name=name, prefix=prefix)
        try:
            data = self._run(args)
            # conda list --json returns a list of packages
            if isinstance(data, list):
                return data
            pkgs = data.get('packages', []) if isinstance(data, dict) else []
            if isinstance(pkgs, list):
                return pkgs
            return []
        except CondaCommandError as e:
            # Fallback: run without JSON and parse the text output
            try:
                res = self._run(args, use_json=False)
                text = (res or {}).get('stdout', '')
                records = self._parse_list_plaintext(text)
                if records:
                    return records
            except Exception:
                pass
            # Re-raise original error if fallback failed
            raise e

    def search(self, term: str, channels: Optional[List[str]] = None) -> Dict[str, Any]:
        args = ['search', term]
        if channels:
            for ch in channels:
                args += ['-c', ch]
        return self._run(args)

    def install(self, pkgs: List[str], prefix: Optional[str] = None, name: Optional[str] = None,
                channels: Optional[List[str]] = None, update_deps: bool = True, yes: bool = True) -> Dict[str, Any]:
        args = ['install'] + self._prefix_args(name=name, prefix=prefix)
        if channels:
            for ch in channels:
                args += ['-c', ch]
        if update_deps:
            args += ['--update-deps']
        if yes:
            args += ['-y']
        args += pkgs
        return self._run(args)

    def update(self, pkgs: Optional[List[str]] = None, prefix: Optional[str] = None, name: Optional[str] = None,
               all_: bool = False, channels: Optional[List[str]] = None, yes: bool = True) -> Dict[str, Any]:
        # If no specific packages are provided, update all by default (UI expectation)
        if pkgs is None:
            all_ = True
        args = ['update'] + self._prefix_args(name=name, prefix=prefix)
        if channels:
            for ch in channels:
                args += ['-c', ch]
        if yes:
            args += ['-y']
        if all_ and not pkgs:
            args += ['--all']
        if pkgs:
            args += pkgs
        return self._run(args)

    def remove(self, pkgs: Optional[List[str]] = None, prefix: Optional[str] = None, name: Optional[str] = None,
               all_: bool = False, yes: bool = True) -> Dict[str, Any]:
        args = ['remove'] + self._prefix_args(name=name, prefix=prefix)
        if yes:
            args += ['-y']
        if all_:
            args += ['--all']
        if pkgs:
            args += pkgs
        return self._run(args)

    def create(self, name: Optional[str] = None, prefix: Optional[str] = None, pkgs: Optional[List[str]] = None,
               channels: Optional[List[str]] = None, yes: bool = True) -> Dict[str, Any]:
        args = ['create'] + self._prefix_args(name=name, prefix=prefix)
        if yes:
            args += ['-y']
        if channels:
            for ch in channels:
                args += ['-c', ch]
        if pkgs:
            args += pkgs
        return self._run(args)

    def config_show(self) -> Dict[str, Any]:
        return self._run(['config', '--show'])

    def get_channels(self) -> List[str]:
        try:
            cfg = self.config_show()
            ch = cfg.get('channels')
            if isinstance(ch, list):
                return ch
        except Exception:
            pass
        # Fallback to defaults
        return ['conda-forge', 'defaults']


# --- Compatibility adapter methods for CondaManagerDialog/UI ---
    def current_prefix(self) -> str:
        """Return the currently active conda prefix, or sys.prefix as fallback."""
        return os.environ.get('CONDA_PREFIX') or sys.prefix

    def list_envs(self) -> List[str]:
        """Return a simple list of environment prefixes (paths)."""
        data = self.get_envs()
        envs = data.get('envs', []) if isinstance(data, dict) else []
        if not isinstance(envs, list):
            return []
        return envs

    def list_installed(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias of list_packages for UI compatibility."""
        return self.list_packages(prefix=prefix)

    def create_env(
        self,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        pkgs: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new environment by name or prefix, optionally with packages."""
        return self.create(name=name, prefix=prefix, pkgs=pkgs, channels=channels, yes=True)

    def remove_env(self, name: Optional[str] = None, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Remove an environment (equivalent to `conda remove --all`)."""
        return self.remove(pkgs=None, prefix=prefix, name=name, all_=True, yes=True)

    def export_env(self, prefix: Optional[str] = None, name: Optional[str] = None) -> str:
        """Export environment to YAML text using `conda env export`."""
        args: List[str] = ['env', 'export']
        if prefix:
            args += ['--prefix', prefix]
        elif name:
            args += ['--name', name]
        res = self._run(args, use_json=False)
        return (res or {}).get('stdout', '')

    def import_env(self, file_path: str, name: Optional[str] = None) -> str:
        """Create an environment from an environment.yml file using `conda env create`."""
        args: List[str] = ['env', 'create', '--file', file_path]
        if name:
            args += ['--name', name]
        res = self._run(args, use_json=False)
        return (res or {}).get('stdout', '')

    def clone_env(
        self,
        name_src: Optional[str] = None,
        prefix_src: Optional[str] = None,
        name_dst: Optional[str] = None,
        prefix_dst: Optional[str] = None,
    ) -> str:
        """Clone an environment to a new name or prefix using `conda create --clone`.
        Returns plain text output for logging in the UI.
        """
        args: List[str] = ['create', '--clone']
        if prefix_src:
            args.append(prefix_src)
        elif name_src:
            args.append(name_src)
        else:
            raise CondaCommandError('Source environment not specified (name or prefix required).')
        if prefix_dst:
            args += ['--prefix', prefix_dst]
        elif name_dst:
            args += ['--name', name_dst]
        else:
            raise CondaCommandError('Destination environment not specified (name or prefix required).')
        args += ['-y']
        res = self._run(args, use_json=False)
        return (res or {}).get('stdout', '')

    def add_channel(self, channel: str) -> str:
        """Add a conda channel using `conda config --add channels <channel>`.
        Returns CLI text output for UI logging.
        """
        res = self._run(['config', '--add', 'channels', channel], use_json=False)
        return (res or {}).get('stdout', '')

    def remove_channel(self, channel: str) -> str:
        """Remove a conda channel using `conda config --remove channels <channel>`.
        Returns CLI text output for UI logging.
        """
        res = self._run(['config', '--remove', 'channels', channel], use_json=False)
        return (res or {}).get('stdout', '')
