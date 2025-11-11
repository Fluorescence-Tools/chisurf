import os
import sys
import json
import subprocess
import platform
import re
import shutil
import pathlib
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

# Path helper to locate user settings folder for CONDARC
from chisurf.settings.path_utils import get_path

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

    New in this version:
    - Prefer using `mamba` for package operations (install/update/remove/list/search/create),
      with automatic fallback to `conda` when `mamba` is not available or fails.
    - Always use a user-writable `.condarc` located inside the ChiSurf user settings
      folder (e.g. `~/.chisurf/.condarc`) to prevent PermissionError when the app
      runs from Program Files. The `CONDARC` env var is set for every subprocess call.
    """

    def __init__(self, conda_executable: Optional[str] = None):
        self._system = platform.system().lower()
        if conda_executable:
            self.conda_exe = conda_executable
        else:
            # Leverage existing discovery logic
            self.conda_exe = ChiSurfUpdater(update_url=None)._get_conda_executable()
        # Discover micromamba
        self.micromamba_exe = self._discover_micromamba()
        # Enforce micromamba-only operation: if no micromamba, fail early (no conda/mamba fallback)
        if not self.micromamba_exe:
            raise CondaCommandError(
                "Micromamba executable not found. Please install micromamba and ensure it is on PATH. Fallback to conda/mamba is disabled.")
        # Allow toggling preference if needed (kept for API compatibility, but must be True)
        self.prefer_mamba: bool = True
        # Ensure a user-level condarc exists and remember its path
        self._condarc_path = self._ensure_user_condarc()

    # --- Internal helpers -------------------------------------------------
    def _ensure_user_condarc(self) -> str:
        """
        Ensure a user-writable condarc file exists in the ChiSurf settings folder.
        Returns the absolute path to the condarc file and guarantees its parent
        directory exists. Handles legacy cases where ".condarc" was created as a
        directory containing a "config" file.
        """
        settings_dir = get_path('settings')
        # Legacy: some installers created a directory named ".condarc" with a file "config" inside
        legacy_dir = settings_dir / '.condarc'
        if legacy_dir.exists() and legacy_dir.is_dir():
            condarc_file = legacy_dir / 'config'
        else:
            condarc_file = settings_dir / '.condarc'
        # Make sure parent exists
        condarc_file.parent.mkdir(parents=True, exist_ok=True)
        # If nothing exists yet, write a minimal safe configuration
        if not condarc_file.exists():
            try:
                condarc_file.write_text(
                    'channels:\n'
                    '  - conda-forge\n'
                    '  - defaults\n'
                    'ssl_verify: true\n',
                    encoding='utf-8'
                )
            except Exception:
                # If we fail to write, still return the path; subprocess will surface errors
                pass
        return str(pathlib.Path(condarc_file).resolve())

    def _base_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = os.environ.copy()
        if self._condarc_path:
            env['CONDARC'] = self._condarc_path
        # Disable conda/mamba plugin system to avoid crashes like "'<pkg>' is not in list"
        # seen with certain conda plugins. We want predictable, fast, mamba-only behavior.
        env['CONDA_NO_PLUGINS'] = 'true'
        # Reduce noise from mamba banner if supported
        env.setdefault('MAMBA_NO_BANNER', '1')
        # Also ensure root prefixes point to a user-writable location under ChiSurf settings
        try:
            settings_dir = get_path('settings')
            root_dir = pathlib.Path(settings_dir) / 'conda-root'
            root_dir.mkdir(parents=True, exist_ok=True)
            root = str(root_dir.resolve())
            env.setdefault('CONDA_ROOT_PREFIX', root)
            env.setdefault('MAMBA_ROOT_PREFIX', root)
        except Exception:
            pass
        if extra:
            env.update(extra)
        return env

    def _wrap_cmd_for_windows(self, cmd: List[str]) -> List[str]:
        if self._system == 'windows':
            exe = (cmd[0] if cmd else '').lower()
            if exe.endswith('.bat') or exe.endswith('.cmd'):
                return ['cmd.exe', '/C', *cmd]
        return cmd

    def _discover_micromamba(self) -> Optional[str]:
        """Find the `micromamba` executable if available on PATH or near conda.
        Returns the absolute path or None if not found.
        """
        # Try PATH first
        for exe_name in ['micromamba', 'micromamba.exe']:
            path = shutil.which(exe_name)
            if path:
                return path
        # Try alongside conda executable (same directory)
        try:
            conda_dir = os.path.dirname(self.conda_exe)
            candidates = [
                os.path.join(conda_dir, 'micromamba'),
                os.path.join(conda_dir, 'micromamba.exe'),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    return c
        except Exception:
            pass
        return None

    def _needs_elevation(self) -> bool:
        """
        Determine if elevated privileges are needed for conda operations.

        Returns:
            Boolean indicating if elevated privileges are needed
        """
        if self._system == "windows":
            # Check if the installation directory is in Program Files
            chisurf_path = get_path('chisurf')
            program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
            program_files_x86 = os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')

            return (str(chisurf_path).startswith(program_files) or
                    str(chisurf_path).startswith(program_files_x86))

        # On Unix-like systems, check if the conda environment is in a system directory
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        return conda_prefix.startswith('/usr') and not conda_prefix.startswith('/usr/local')

    def _run_with_exe(self, exe: str, args: List[str], use_json: bool = True, env: Optional[Dict[str, str]] = None, on_progress: Optional[Callable[[str], None]] = None):
        cmd = [exe, *args]
        # Prefer quiet mode to reduce non-JSON noise
        if '--quiet' not in cmd and '-q' not in cmd:
            cmd.append('--quiet')
        if use_json and '--json' not in cmd:
            cmd.append('--json')
        popen_cmd = self._wrap_cmd_for_windows(cmd)
        try:
            stream = on_progress is not None
            proc = subprocess.Popen(
                popen_cmd,
                stdout=subprocess.PIPE,
                stderr=(subprocess.STDOUT if stream else subprocess.PIPE),
                text=True,
                encoding='utf-8',
                errors='replace',
                shell=False,
                env=self._base_env(env),
            )
            if stream:
                collected: List[str] = []
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        if not line:
                            break
                        collected.append(line)
                        try:
                            on_progress(line.rstrip('\r\n'))
                        except Exception:
                            pass
                finally:
                    proc.wait()
                stdout = ''.join(collected)
                stderr = ''
            else:
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
                    msg = (stderr or '').strip() or f"Command failed: {os.path.basename(exe)}"
                raise CondaCommandError(msg, self._truncate(stdout), self._truncate(stderr), proc.returncode)
            if use_json:
                # Strict first
                try:
                    data = json.loads(stdout or '{}')
                except Exception:
                    # Relaxed parsing
                    try:
                        data = self._parse_json_relaxed(stdout)
                    except Exception as e:
                        raise CondaCommandError(f"Invalid JSON from {os.path.basename(exe)}: {e}\nOutput: {self._truncate(stdout)}",
                                                self._truncate(stdout), self._truncate(stderr), proc.returncode)
                # Unwrap micromamba envelope { success: bool, result: ... }
                if isinstance(data, dict) and 'result' in data and ('success' in data or 'status' in data):
                    return data.get('result')
                return data
            else:
                # Return plain text wrapped in a dict
                return {"stdout": stdout, "stderr": stderr}
        except CondaCommandError:
            raise
        except Exception as e:
            raise CondaCommandError(f"Failed to run {os.path.basename(exe)}: {e}")

    def _run_micromamba_first(self, args: List[str], use_json: bool = True, env: Optional[Dict[str, str]] = None, on_progress: Optional[Callable[[str], None]] = None):
        """Run using micromamba only. Fallback has been removed by design."""
        if not (self.prefer_mamba and self.micromamba_exe):
            raise CondaCommandError("Micromamba is not available, and fallback is disabled.")
        return self._run_with_exe(self.micromamba_exe, args, use_json=use_json, env=env, on_progress=on_progress)

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

    def _run(self, args: List[str], use_json: bool = True, env: Optional[Dict[str, str]] = None, on_progress: Optional[Callable[[str], None]] = None):
        # Enforce micromamba usage for all operations
        exe = self.micromamba_exe or 'micromamba'
        # Micromamba does not use conda's plugin system; avoid passing --no-plugins
        cmd = [exe, *args]
        # Prefer quiet mode to reduce non-JSON noise
        if '--quiet' not in cmd and '-q' not in cmd:
            cmd.append('--quiet')
        if use_json and '--json' not in cmd:
            cmd.append('--json')
        popen_cmd = self._wrap_cmd_for_windows(cmd)
        try:
            stream = on_progress is not None
            proc = subprocess.Popen(
                popen_cmd,
                stdout=subprocess.PIPE,
                stderr=(subprocess.STDOUT if stream else subprocess.PIPE),
                text=True,
                encoding='utf-8',
                errors='replace',
                shell=False,
                env=self._base_env(env),
            )
            if stream:
                collected: List[str] = []
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        if not line:
                            break
                        collected.append(line)
                        try:
                            on_progress(line.rstrip('\r\n'))
                        except Exception:
                            pass
                finally:
                    proc.wait()
                stdout = ''.join(collected)
                stderr = ''
            else:
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
                    msg = (stderr or '').strip() or 'Micromamba command failed'
                raise CondaCommandError(msg, self._truncate(stdout), self._truncate(stderr), proc.returncode)
            if use_json:
                # Strict first
                try:
                    data = json.loads(stdout or '{}')
                except Exception:
                    # Relaxed parsing
                    try:
                        data = self._parse_json_relaxed(stdout)
                    except Exception as e:
                        raise CondaCommandError(f"Invalid JSON from micromamba: {e}\nOutput: {self._truncate(stdout)}",
                                                self._truncate(stdout), self._truncate(stderr), proc.returncode)
                # Unwrap micromamba envelope { success: bool, result: ... }
                if isinstance(data, dict) and 'result' in data and ('success' in data or 'status' in data):
                    return data.get('result')
                return data
            else:
                # Return plain text wrapped in a dict
                return {"stdout": stdout, "stderr": stderr}
        except CondaCommandError:
            raise
        except Exception as e:
            raise CondaCommandError(f"Failed to run micromamba: {e}")

    def _prefix_args(self, name: Optional[str] = None, prefix: Optional[str] = None) -> List[str]:
        args: List[str] = []
        if prefix:
            args += ['--prefix', prefix]
        elif name:
            args += ['--name', name]
        return args

    def _run_with_fallback(self, args: List[str], env: Optional[Dict[str, str]] = None, on_progress: Optional[Callable[[str], None]] = None):
        """Run a micromamba command with JSON first, and retry without JSON on CondaCommandError.
        Returns parsed JSON (dict/list) or a dict with stdout/stderr from the non-JSON retry.
        """
        try:
            return self._run(args, use_json=True, env=env, on_progress=on_progress)
        except CondaCommandError:
            return self._run(args, use_json=False, env=env, on_progress=on_progress)

    def _run_micromamba_first_with_fallback(self, args: List[str], env: Optional[Dict[str, str]] = None, on_progress: Optional[Callable[[str], None]] = None):
        """Run with micromamba-only using JSON; on CondaCommandError retry without JSON."""
        try:
            return self._run_micromamba_first(args, use_json=True, env=env, on_progress=on_progress)
        except CondaCommandError:
            return self._run_micromamba_first(args, use_json=False, env=env, on_progress=on_progress)

    # --- Recovery helpers --------------------------------------------------
    def _contains_any(self, hay: str, needles: List[str]) -> bool:
        hay = (hay or "").lower()
        for n in needles:
            if n.lower() in hay:
                return True
        return False

    def _clean_package_cache(self, on_progress: Optional[Callable[[str], None]] = None) -> None:
        """Attempt to clean broken/corrupted packages from the cache using micromamba.
        Uses: micromamba clean --packages --tarballs -y (no JSON).
        """
        try:
            if on_progress:
                on_progress("[Micromamba] Cleaning local package cache (micromamba clean --packages --tarballs)…")
            self._run(['clean', '--packages', '--tarballs', '-y'], use_json=False, on_progress=on_progress)
            if on_progress:
                on_progress("[Micromamba] Package cache cleaned.")
        except Exception as _:
            # Non-fatal: ignore, we'll still retry the operation
            if on_progress:
                on_progress("[Micromamba] Cache clean did not complete (ignored).")

    def _run_with_recovery(self, args: List[str], on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Execute install-like commands with mamba-only and minimal recovery strategy.
        1) Try micromamba with JSON/plain fallback.
        2) If we see SafetyError/Verification/Clobber errors, clean cache and retry once with micromamba.
        """
        # First try normal path
        try:
            return self._run_micromamba_first_with_fallback(args, on_progress=on_progress)
        except CondaCommandError as e:
            msg = str(e)
            # Mitigation for mamba/conda plugin crashes like "'<pkg>' is not in list" or plugin stack traces.
            if self._contains_any(msg, ["is not in list", "plugin", "--no-plugins"]):
                if on_progress:
                    on_progress("[Micromamba] Detected plugin-related crash. Retrying once without plugins and with plain output…")
                try:
                    # Micromamba doesn't use plugins; just retry without JSON
                    patched_args = args[:]
                    return self._run_micromamba_first(patched_args, use_json=False, on_progress=on_progress)
                except CondaCommandError:
                    # Fall through to other recovery checks
                    pass
            # If mamba reported cache/collision errors, try clean + mamba retry once
            if self._contains_any(msg, ["safetyerror", "verificationerror", "clobbererror", "condaverificationerror", "unknownpackageclobbererror"]):
                if on_progress:
                    on_progress("[Micromamba] Detected corrupted cache or path collision. Cleaning cache and retrying with micromamba…")
                self._clean_package_cache(on_progress=on_progress)
                return self._run_micromamba_first_with_fallback(args, on_progress=on_progress)
            # Unknown error: rethrow
            raise

    def _run_with_elevation(self, cmd: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Run a command with elevated privileges on Windows.

        Args:
            cmd: Command to run as a list of arguments

        Returns:
            Tuple containing:
            - Boolean indicating if the command was successful
            - Error message if the command failed, None otherwise
        """
        if self._system != "windows":
            return self._run_command(cmd)

        try:
            # Create a temporary batch file to run the command with logging
            temp_dir = tempfile.mkdtemp(prefix="chisurf_conda_elev_")
            log_file = os.path.join(temp_dir, "elevated_conda_command.log")

            # Properly quote arguments that contain spaces
            quoted_cmd = [f'"{arg}"' if ' ' in str(arg) and not str(arg).startswith('"') else str(arg) for arg in cmd]
            win_cmd_str = " ".join(quoted_cmd)

            batch_file = os.path.join(temp_dir, "run_conda_elevated.bat")
            with open(batch_file, 'w') as f:
                f.write('@echo off\n')
                f.write(f'echo Running elevated conda command at %DATE% %TIME% > "{log_file}"\n')
                f.write(f'echo Command: {win_cmd_str} >> "{log_file}"\n')
                f.write('echo. >> "' + log_file + '"\n')
                # Execute the command and capture all output to the log
                f.write(f'{win_cmd_str} >> "{log_file}" 2>&1\n')
                f.write('set EXITCODE=%ERRORLEVEL%\n')
                f.write('echo. >> "' + log_file + '"\n')
                f.write('echo Exit code: %EXITCODE% >> "' + log_file + '"\n')
                f.write('if %EXITCODE% NEQ 0 (\n')
                f.write('  echo Elevated conda command failed with error code %EXITCODE% >> "' + log_file + '"\n')
                f.write('  exit /b %EXITCODE%\n')
                f.write(')\n')
                f.write('echo Elevated conda command completed successfully >> "' + log_file + '"\n')
                f.write('exit /b 0\n')

            # Run the batch file with elevated privileges using PowerShell and wait for completion
            powershell_cmd = [
                'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
                f"$p = Start-Process -FilePath '{batch_file}' -Verb RunAs -Wait -PassThru; exit $p.ExitCode"
            ]

            logging.debug(f"Running elevated conda batch: {batch_file}")
            logging.debug(f"Elevated conda log will be written to: {log_file}")

            process = subprocess.Popen(
                powershell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if stdout:
                logging.debug(f"Elevation launcher stdout:\n{stdout}")
            if stderr:
                logging.debug(f"Elevation launcher stderr:\n{stderr}")

            # Read last lines of the elevated log if present for quick context
            tail_hint = ""
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r', errors='ignore') as lf:
                        lines = lf.readlines()
                        tail = "".join(lines[-25:]) if lines else ""
                        tail_hint = tail.strip()
            except Exception:
                pass

            if process.returncode != 0:
                err_msg = f"Elevation failed with exit code {process.returncode}. See log: {log_file}"
                if tail_hint:
                    err_msg += f"\n--- Log tail ---\n{tail_hint}"
                return False, err_msg

            # The batch itself exits with the wrapped command's exit code; inspect the log tail for visibility
            logging.info(f"Elevated conda command finished. Log: {log_file}")
            if tail_hint:
                logging.debug(f"Elevated conda command log tail:\n{tail_hint}")

            return True, None
        except Exception as e:
            return False, str(e)

    # --- Public API -------------------------------------------------------
    def info(self, on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        return self._run(['info'], on_progress=on_progress)

    def get_envs(self, on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
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

    def list_packages(self, prefix: Optional[str] = None, name: Optional[str] = None, on_progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
        args = ['list'] + self._prefix_args(name=name, prefix=prefix)
        try:
            data = self._run(args, on_progress=on_progress)
            # Normalize various shapes to a list of records
            pkgs: List[Dict[str, Any]] = []
            if isinstance(data, list):
                pkgs = data
            elif isinstance(data, dict):
                # micromamba typically returns {packages:[...]}, older conda: list directly
                if 'packages' in data and isinstance(data['packages'], list):
                    pkgs = data['packages']
                elif 'result' in data and isinstance(data['result'], list):
                    pkgs = data['result']
            # Enrich with derived fields
            return [self._enrich_record(rec) for rec in pkgs if isinstance(rec, dict)]
        except CondaCommandError as e:
            # Fallback: run without JSON and parse the text output
            try:
                res = self._run(args, use_json=False)
                text = (res or {}).get('stdout', '')
                records = self._parse_list_plaintext(text)
                if records:
                    return [self._enrich_record(r) for r in records]
            except Exception:
                pass
            # Re-raise original error if fallback failed
            raise e

    def search(self, term: str, channels: Optional[List[str]] = None, on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        args = ['search', term]
        if channels:
            for ch in channels:
                args += ['-c', ch]
        # Prefer micromamba for search too; use non-JSON fallback on error
        raw = self._run_micromamba_first_with_fallback(args, on_progress=on_progress)
        # Normalize various possible outputs to a consistent {"packages": [records...]}
        records: List[Dict[str, Any]] = []

        def _add_from_list(lst: List[Any], name_hint: Optional[str] = None):
            for r in lst:
                if isinstance(r, dict):
                    if name_hint and 'name' not in r:
                        r['name'] = name_hint
                    records.append(self._enrich_record(r))

        if isinstance(raw, list):
            # Some variants may return flat list
            _add_from_list(raw)
        elif isinstance(raw, dict):
            # Newer micromamba shape: {"query": {...}, "result": {"pkgs": [ ... ]}} or {"result": [ ... ]}
            res = raw.get('result') if 'result' in raw else None
            if isinstance(res, dict):
                if isinstance(res.get('pkgs'), list):
                    _add_from_list(res['pkgs'])
                else:
                    # Some tools may put name-keys under result
                    for k, v in res.items():
                        if isinstance(v, list):
                            _add_from_list(v, name_hint=k)
            elif isinstance(res, list):
                _add_from_list(res)
            # Classic mamba/conda JSON
            if not records:
                # micromamba can return {packages:{name:[...]}} or {packages:[...]}
                pkgs = raw.get('packages')
                if isinstance(pkgs, dict):
                    for name, recs in pkgs.items():
                        if isinstance(recs, list):
                            _add_from_list(recs, name_hint=name)
                elif isinstance(pkgs, list):
                    _add_from_list(pkgs)
                else:
                    # Attempt other top-level shapes {name:[...]}
                    for k, v in raw.items():
                        if isinstance(v, list):
                            _add_from_list(v, name_hint=k)
        return {"packages": records}

    def install(self, pkgs: List[str], prefix: Optional[str] = None, name: Optional[str] = None,
               channels: Optional[List[str]] = None, update_deps: bool = True, yes: bool = True,
               on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        args = ['install'] + self._prefix_args(name=name, prefix=prefix)
        if channels:
            for ch in channels:
                args += ['-c', ch]
        # Note: micromamba (our default runner) may not support --update-deps; omit to ensure compatibility.
        # Conda generally updates dependencies as needed during install; if a future need arises,
        # consider capability detection against `install --help` and add the flag conditionally.
        if yes:
            args += ['-y']
        args += pkgs

        # Check if elevated privileges are needed
        needs_elevation = self._needs_elevation()
        if needs_elevation:
            logging.info("Elevated privileges required for conda install operation")
            if on_progress:
                on_progress("Administrator privileges are required for this operation.")
            # Use elevated execution
            success, error_msg = self._run_with_elevation([self.micromamba_exe] + args)
            if not success:
                raise CondaCommandError(f"Install failed with elevated privileges: {error_msg}")
            # Return a success result similar to _run_with_recovery format
            return {"success": True, "message": "Package(s) installed successfully with elevated privileges"}
        else:
            # Prefer micromamba with robust recovery (handles cache/collision errors)
            return self._run_with_recovery(args, on_progress=on_progress)

    def update(self, pkgs: Optional[List[str]] = None, prefix: Optional[str] = None, name: Optional[str] = None,
               all_: bool = False, channels: Optional[List[str]] = None, yes: bool = True,
               on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
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

        # Check if elevated privileges are needed
        needs_elevation = self._needs_elevation()
        if needs_elevation:
            logging.info("Elevated privileges required for conda update operation")
            if on_progress:
                on_progress("Administrator privileges are required for this operation.")
            # Use elevated execution
            success, error_msg = self._run_with_elevation([self.micromamba_exe] + args)
            if not success:
                raise CondaCommandError(f"Update failed with elevated privileges: {error_msg}")
            # Return a success result similar to _run_with_recovery format
            return {"success": True, "message": "Package(s) updated successfully with elevated privileges"}
        else:
            # Prefer micromamba with robust recovery
            return self._run_with_recovery(args, on_progress=on_progress)

    def remove(self, pkgs: Optional[List[str]] = None, prefix: Optional[str] = None, name: Optional[str] = None,
               all_: bool = False, yes: bool = True, on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        args = ['remove'] + self._prefix_args(name=name, prefix=prefix)
        if yes:
            args += ['-y']
        if all_:
            args += ['--all']
        if pkgs:
            args += pkgs

        # Check if elevated privileges are needed
        needs_elevation = self._needs_elevation()
        if needs_elevation:
            logging.info("Elevated privileges required for conda remove operation")
            if on_progress:
                on_progress("Administrator privileges are required for this operation.")
            # Use elevated execution
            success, error_msg = self._run_with_elevation([self.micromamba_exe] + args)
            if not success:
                raise CondaCommandError(f"Remove failed with elevated privileges: {error_msg}")
            # Return a success result similar to _run_with_recovery format
            return {"success": True, "message": "Package(s) removed successfully with elevated privileges"}
        else:
            # Prefer micromamba with robust recovery
            return self._run_with_recovery(args, on_progress=on_progress)

    def create(self, name: Optional[str] = None, prefix: Optional[str] = None, pkgs: Optional[List[str]] = None,
               channels: Optional[List[str]] = None, yes: bool = True, on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        args = ['create'] + self._prefix_args(name=name, prefix=prefix)
        if yes:
            args += ['-y']
        if channels:
            for ch in channels:
                args += ['-c', ch]
        if pkgs:
            args += pkgs
        return self._run(args, on_progress=on_progress)

    def config_show(self, on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Return configuration as a dict. Prefer JSON if micromamba supports it."""
        # Try micromamba JSON first
        try:
            data = self._run(['config', 'list', '--json'], on_progress=on_progress)
            if isinstance(data, dict):
                cfg = data.get('rc', data) if 'rc' in data else data
            else:
                cfg = {}
            # Attach detected rc path for UI convenience
            cfg.setdefault('rc_path', self._condarc_path)
            return cfg
        except Exception:
            pass
        # Fallback to plaintext show
        plain = self._run(['config', '--show'], use_json=False, on_progress=on_progress)
        text = (plain or {}).get('stdout', '')
        cfg: Dict[str, Any] = {}
        channels: List[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.lower().startswith('channels:'):
                # Next lines with leading '-' are channels
                continue
            if s.startswith('- '):
                ch = s[2:].strip()
                if ch:
                    channels.append(ch)
            else:
                # simple key: value
                if ':' in s:
                    k, v = s.split(':', 1)
                    cfg[k.strip()] = v.strip()
        if channels:
            cfg['channels'] = channels
        cfg.setdefault('rc_path', self._condarc_path)
        return cfg

    def add_channel(self, channel: str) -> Dict[str, Any]:
        """Add a channel to conda configuration.
        Strategy (robust across micromamba/conda variations):
          1) If channel already present, return success immediately.
          2) Try micromamba-native: `config append channels <channel>`.
          3) Try alternative micromamba form: `config prepend channels <channel>` (keeps higher priority).
          4) Try conda-style: `config --add channels <channel>` (only works on conda/mamba, not micromamba).
          5) As last resort, edit .condarc YAML in place to add the channel.
        Returns a dict describing the method used.
        """
        ch = (channel or '').strip()
        if not ch:
            return {'ok': True, 'method': 'noop', 'reason': 'empty-channel'}

        # 1) Already present?
        try:
            existing = self.get_channels()
            if isinstance(existing, list) and ch in existing:
                return {'ok': True, 'method': 'already', 'channels': existing}
        except Exception:
            # Continue attempts
            pass

        # 2) micromamba: append
        try:
            res = self._run_micromamba_first_with_fallback(['config', 'append', 'channels', ch])
            # Verify it took effect
            try:
                now = self.get_channels()
                if isinstance(now, list) and ch in now:
                    return {'ok': True, 'method': 'micromamba-append', 'channels': now}
            except Exception:
                pass
        except Exception:
            pass

        # 3) micromamba: prepend (places channel earlier)
        try:
            res = self._run_micromamba_first_with_fallback(['config', 'prepend', 'channels', ch])
            try:
                now = self.get_channels()
                if isinstance(now, list) and ch in now:
                    return {'ok': True, 'method': 'micromamba-prepend', 'channels': now}
            except Exception:
                pass
        except Exception:
            pass

        # 4) conda-style `--add` (only works when underlying runner supports it)
        try:
            res = self._run_with_fallback(['config', '--add', 'channels', ch])
            try:
                now = self.get_channels()
                if isinstance(now, list) and ch in now:
                    return {'ok': True, 'method': 'conda-add', 'channels': now}
            except Exception:
                pass
        except Exception:
            pass

        # 5) YAML fallback edit of .condarc
        try:
            return self._add_channel_via_yaml(ch)
        except Exception as e:
            raise CondaCommandError(f"Failed to add channel '{ch}': {e}")

    def _add_channel_via_yaml(self, channel: str) -> Dict[str, Any]:
        """Safely add a channel to the condarc YAML as a last resort.
        This does not execute any conda command; it edits the rc file directly.
        """
        import io
        try:
            import yaml  # PyYAML
        except Exception as e:
            raise CondaCommandError(f"PyYAML not available to edit condarc: {e}")

        rc_path = self._condarc_path
        # Ensure parent directory exists
        try:
            pathlib.Path(rc_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        data: Dict[str, Any] = {}
        if os.path.exists(rc_path):
            try:
                with open(rc_path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)  # may be None
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception:
                # if YAML invalid, start fresh minimal config
                data = {}

        channels = data.get('channels')
        if not isinstance(channels, list):
            channels = []
        if channel not in channels:
            channels.append(channel)
        data['channels'] = channels

        # Write back atomically (best-effort on Windows)
        tmp_path = rc_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        try:
            # Replace original
            if os.path.exists(rc_path):
                try:
                    os.replace(tmp_path, rc_path)
                except Exception:
                    # Fallback to remove+rename
                    try:
                        os.remove(rc_path)
                    except Exception:
                        pass
                    os.rename(tmp_path, rc_path)
            else:
                os.rename(tmp_path, rc_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        # Verify via config_show
        now = self.get_channels()
        if isinstance(now, list) and channel in now:
            return {'ok': True, 'method': 'yaml', 'channels': now, 'rc_path': rc_path}
        # Even if config_show didn't reflect it, return success with file path
        return {'ok': True, 'method': 'yaml', 'channels': now, 'rc_path': rc_path}

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

    # --- Helpers for UI enrichment ----------------------------------------
    def condarc_path(self) -> str:
        """Expose the condarc path used for subprocess calls."""
        return self._condarc_path or ''

    def _enrich_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure standard fields exist and derive repository/channel information.
        Adds:
          - channel: normalized channel name (e.g., 'conda-forge' instead of 't/<TOKEN>/conda-forge')
          - base_url: kept if present
          - repo: human-friendly source (usually same as channel)
        """
        import re
        r = dict(rec) if isinstance(rec, dict) else {}
        base_url = r.get('base_url') or r.get('url_base')

        # Prefer explicit channel from record
        raw_channel = r.get('channel') or ''

        # If no channel present, try to infer from base_url
        if not raw_channel and base_url and isinstance(base_url, str):
            try:
                # e.g. https://conda.anaconda.org/conda-forge
                raw_channel = base_url.rstrip('/').split('/')[-1]
            except Exception:
                raw_channel = ''

        # As a last resort, fall back to subdir/platform (better than empty)
        if not raw_channel:
            raw_channel = r.get('subdir') or r.get('platform') or 'unknown'

        # Normalize tokenized/URL-like channel specifications
        ch = str(raw_channel or '').strip()
        # Strip private token prefix pattern like "t/<TOKEN>/"
        ch = re.sub(r"^t/[^/]+/", "", ch)
        # If the remaining channel still contains slashes, keep the last non-empty segment
        if '/' in ch:
            parts = [p for p in ch.split('/') if p]
            if parts:
                ch = parts[-1]

        # If base_url clearly indicates conda-forge, force channel to 'conda-forge'
        if isinstance(base_url, str) and 'conda-forge' in base_url:
            ch = 'conda-forge'

        # A few well-known normalizations
        mapped = {
            'conda forge': 'conda-forge',
            'defaults': 'defaults',
            'pypi': 'pypi',
        }
        ch_l = ch.lower()
        if ch_l in mapped:
            ch = mapped[ch_l]

        r['channel'] = ch

        # For UI: repo is just the normalized channel (avoid leaking URLs/tokens)
        r.setdefault('repo', ch)

        # Standardize build string key
        if 'build' in r and 'build_string' not in r:
            r['build_string'] = r['build']
        return r


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

    def list_installed(self, prefix: Optional[str] = None, on_progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
        """Alias of list_packages for UI compatibility."""
        return self.list_packages(prefix=prefix, on_progress=on_progress)

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

    # NOTE:
    # A robust `add_channel(self, channel: str)` implementation already exists above
    # (lines ~698-762). Do NOT re‑declare it here, otherwise the Python class will
    # override the robust implementation with this simplified variant.
    # The previous minimal version that called `config --add` unconditionally has
    # been removed to ensure micromamba compatibility and to allow YAML fallback.

    def remove_channel(self, channel: str) -> str:
        """Remove a conda channel using `conda config --remove channels <channel>`.
        Returns CLI text output for UI logging.
        """
        res = self._run(['config', '--remove', 'channels', channel], use_json=False)
        return (res or {}).get('stdout', '')
