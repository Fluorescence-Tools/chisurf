"""ChiMol interactive CLI powered by ptpython.

Provides a rich REPL that supports both chimol commands (``load``,
``fetch``, ``color``, …) and arbitrary Python expressions in a single
prompt.  When ptpython is not installed the CLI gracefully falls back
to a bare ``input()`` loop.

This module is intended as a **blueprint** for eventually replacing
the ``qtconsole`` dependency across all of ChiSurf.

Usage::

    python -m chisurf.plugins.chimol cli            # interactive REPL
    python -m chisurf.plugins.chimol cli -e "fetch 1crn; objects"
    python -m chisurf.plugins.chimol cli -s demo.pml
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# History helpers (shared between ptpython and fallback paths)
# ---------------------------------------------------------------------------

def _resolve_history_path() -> Path:
    """Return a platform-appropriate path for chimol CLI history."""
    try:
        import chisurf.core.settings as _cs_settings
        base = _cs_settings.get_path("settings")
        return Path(base) / "chimol_cli_history"
    except Exception:
        return Path.home() / ".chimol_cli_history"


# ---------------------------------------------------------------------------
# Chimol Cmd bootstrap (Qt-free)
# ---------------------------------------------------------------------------

def _make_cmd():
    """Create a headless ``Cmd`` instance backed by a mock viewer."""
    from ..testing.mock_viewer import MockWindow
    from ..cmd.command import Cmd

    win = MockWindow()
    cmd = Cmd(win)
    cmd.set_message_callback(lambda msg: print(f"\033[36m{msg}\033[0m"))
    cmd.set_error_callback(
        lambda err: print(f"\033[31m[error]\033[0m {err}", file=sys.stderr)
    )
    return cmd


# ---------------------------------------------------------------------------
# prompt-toolkit / ptpython completer for chimol commands
# ---------------------------------------------------------------------------

def _build_chimol_completer(cmd_instance):
    """Return a prompt-toolkit ``Completer`` that knows chimol commands."""
    from prompt_toolkit.completion import Completer, Completion
    import re

    class ChimolCompleter(Completer):
        _REP_NAMES = [
            "cartoon", "sticks", "atoms", "dots", "surface",
            "ca_trace", "lines", "spheres", "metaball", "plane", "all",
        ]
        _COLOR_NAMES = [
            "red", "green", "blue", "yellow", "cyan", "magenta",
            "white", "black", "gray", "orange",
            "single", "by_residue", "by_ss", "by_sequence",
            "byelement", "bychain", "spectrum",
        ]
        _SETTING_NAMES = [
            "bg_color", "color_mode", "line_width", "stick_radius",
            "sphere_scale", "cartoon_transparency", "surface_type",
        ]

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor.lstrip()
            if not text:
                return

            # Split the line into tokens, respecting spaces and commas
            # We want to find the "active" token being typed
            parts = re.split(r"[\s,]+", text)
            
            # 1. Complete command name (first token)
            if len(parts) <= 1 and not text.endswith((" ", ",")):
                word = parts[0].lower()
                for name in sorted(cmd_instance._commands.keys()):
                    if name.startswith(word):
                        yield Completion(name, start_position=-len(word), display_meta="cmd")
                return

            # 2. Argument completions
            cmd_name = parts[0].lower()
            if cmd_name not in cmd_instance._commands:
                return

            # Determine what pool to use based on command and position
            pool = []
            
            # Current word being typed (if cursor is at the end of a token)
            # or empty string (if cursor is after a separator)
            if text.endswith((" ", ",")):
                current_word = ""
            else:
                current_word = parts[-1].lower()

            # Representation completions
            if cmd_name in {"show", "hide", "as"}:
                pool.extend(self._REP_NAMES)
            
            # Color completions
            if cmd_name in {"color", "bg_color", "bg_colour"}:
                pool.extend(self._COLOR_NAMES)
                
            # Settings completions
            if cmd_name in {"set", "get"}:
                pool.extend(self._SETTING_NAMES)
                pool.extend([
                    "metaball_alpha", "metaball_shininess", "metaball_threshold", 
                    "metaball_resolution", "metaball_radius", "metaball_padding"
                ])

            # Object name completions (dynamic)
            if cmd_name in {"select", "delete", "enable", "disable", "show", "hide", "color", "center", "zoom"}:
                try:
                    _, viewer = cmd_instance._require_window_and_viewer()
                    if viewer is not None:
                        for obj in viewer.list_objects():
                            if "name" in obj:
                                pool.append(obj["name"])
                except Exception:
                    pass

            # Yield matches from the pool
            seen = set()
            for item in sorted(pool):
                if item.lower() in seen:
                    continue
                seen.add(item.lower())
                
                if item.lower().startswith(current_word):
                    yield Completion(
                        item, 
                        start_position=-len(current_word), 
                        display_meta="arg"
                    )

    return ChimolCompleter()


# ---------------------------------------------------------------------------
# ptpython REPL configuration callback
# ---------------------------------------------------------------------------

def _configure_repl(repl, *, cmd_instance) -> None:
    """Configure the embedded ptpython REPL for chimol usage."""
    from ptpython.layout import CompletionVisualisation

    repl.show_signature = True
    repl.show_docstring = False
    repl.show_meta_enter_message = False
    repl.highlight_matching_parenthesis = True
    repl.wrap_lines = True
    repl.complete_while_typing = True
    repl.completion_visualisation = CompletionVisualisation.POP_UP
    repl.enable_fuzzy_completion = True
    repl.enable_auto_suggest = True
    repl.enable_open_in_editor = True
    repl.enable_system_bindings = True
    repl.confirm_exit = False
    repl.enable_input_validation = True
    repl.insert_blank_line_after_output = False
    repl.prompt_style = "classic"
    repl.enable_history_search = False
    repl.enable_mouse_support = False

    # Dark-friendly colour scheme
    repl.use_code_colorscheme("monokai")
    repl.color_depth = "DEPTH_8_BIT"
    repl.min_brightness = 0.15

    # Custom title for the status bar
    repl.title = "ChiMol"


# ---------------------------------------------------------------------------
# Custom input handler: chimol commands vs Python
# ---------------------------------------------------------------------------

def _make_namespace(cmd_instance) -> Dict[str, Any]:
    """Build the namespace dict injected into the REPL.

    This namespace exposes:
    * ``cmd`` — the ``Cmd`` instance (for ``cmd.load(...)``, etc.)
    * Every public Python-level API of ``Cmd`` as a top-level function
      (``load``, ``fetch``, ``color``, …)
    * ``do``  — raw command-line executor (``do("load foo.pdb")``)
    * Common scientific imports (``np``, ``Path``)
    """
    import numpy as _np

    ns: Dict[str, Any] = {
        "cmd": cmd_instance,
        "do": cmd_instance.do,
        "np": _np,
        "Path": Path,
    }

    # Mirror all public methods of the Cmd instance
    for attr_name in dir(cmd_instance):
        if attr_name.startswith("_"):
            continue
        attr = getattr(cmd_instance, attr_name, None)
        if callable(attr) and attr_name not in ns:
            ns[attr_name] = attr

    return ns


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

_BANNER = """\
\033[1;36m╔══════════════════════════════════════════════════════════════╗
║              ChiMol — Interactive Molecular CLI              ║
║                  PyMOL-compatible command layer               ║
╚══════════════════════════════════════════════════════════════╝\033[0m

  \033[33mCommands\033[0m : Type chimol commands directly, e.g. \033[32mfetch 1crn\033[0m
  \033[33mPython\033[0m   : Any valid Python expression also works
  \033[33mHelp\033[0m     : \033[32mhelp()\033[0m  or  \033[32mdo("help")\033[0m
  \033[33mQuit\033[0m     : \033[32mexit()\033[0m  or  Ctrl-D
"""


# ---------------------------------------------------------------------------
# ptpython REPL launcher
# ---------------------------------------------------------------------------

def _run_ptpython_repl(cmd_instance) -> None:
    """Launch the ptpython interactive REPL with chimol extensions."""
    from ptpython.repl import embed, PythonRepl

    ns = _make_namespace(cmd_instance)
    history_path = str(_resolve_history_path())

    completer = _build_chimol_completer(cmd_instance)

    def configure(repl: PythonRepl) -> None:
        _configure_repl(repl, cmd_instance=cmd_instance)
        # Merge chimol completer with the default Python completer
        _install_chimol_completer(repl, completer)
        # Install a key binding that intercepts plain chimol commands
        _install_chimol_keybinding(repl, cmd_instance)

    print(_BANNER)

    embed(
        globals=ns,
        locals=ns,
        history_filename=history_path,
        configure=configure,
        title="ChiMol",
    )


def _install_chimol_completer(repl, chimol_completer) -> None:
    """Merge the chimol ``Completer`` into ptpython's completion pipeline."""
    from prompt_toolkit.completion import merge_completers

    original_completer = repl.completer
    if original_completer is not None:
        repl.completer = merge_completers([chimol_completer, original_completer])
    else:
        repl.completer = chimol_completer


def _install_chimol_keybinding(repl, cmd_instance) -> None:
    """Add a key binding so that bare chimol commands (``load foo.pdb``)
    are intercepted before Python evaluation.

    Strategy: on Enter, if the current buffer text starts with a known
    chimol command name and is NOT valid Python, route it through
    ``cmd.do()`` instead.
    """
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.filters import HasFocus
    from prompt_toolkit.enums import DEFAULT_BUFFER

    bindings = KeyBindings()

    @bindings.add(Keys.Enter, filter=HasFocus(DEFAULT_BUFFER), eager=True)
    def _handle_enter(event):
        buf = event.app.current_buffer
        text = buf.text.strip()

        if not text:
            buf.validate_and_handle()
            return

        first_token = text.split()[0].lower()
        is_chimol_cmd = first_token in cmd_instance._commands

        if is_chimol_cmd:
            # Check whether it is also valid Python
            is_python = False
            try:
                compile(text, "<input>", "eval")
                is_python = True
            except SyntaxError:
                try:
                    compile(text, "<input>", "exec")
                    is_python = True
                except SyntaxError:
                    pass

            if not is_python:
                # Execute as chimol command
                buf.reset()
                try:
                    cmd_instance.do(text)
                except Exception as exc:
                    print(f"\033[31m[error]\033[0m {exc}", file=sys.stderr)
                return

        # Fall through to normal ptpython handling
        buf.validate_and_handle()

    repl.app.key_bindings = merge_key_bindings_safe(
        repl.app.key_bindings, bindings
    )


def merge_key_bindings_safe(existing, extra):
    """Merge key bindings, handling the fact that ptpython may use
    ``_MergedKeyBindings`` internally."""
    from prompt_toolkit.key_binding import merge_key_bindings

    return merge_key_bindings([existing, extra])


# ---------------------------------------------------------------------------
# Fallback REPL (no ptpython)
# ---------------------------------------------------------------------------

def _run_fallback_repl(cmd_instance) -> None:
    """Simple ``input()`` REPL when ptpython is not available."""
    print(_BANNER)
    print("  \033[33m(ptpython not installed — using basic REPL)\033[0m\n")

    ns = _make_namespace(cmd_instance)

    while True:
        try:
            line = input("\033[36mchimol>\033[0m ").strip()
            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                break

            # Try as chimol command first
            first_token = line.split()[0].lower()
            if first_token in cmd_instance._commands:
                cmd_instance.do(line)
                continue

            # Try as Python
            try:
                result = eval(line, ns)
                if result is not None:
                    print(result)
            except SyntaxError:
                try:
                    exec(line, ns)
                except Exception as e:
                    print(f"\033[31m[error]\033[0m {e}", file=sys.stderr)
            except Exception as e:
                print(f"\033[31m[error]\033[0m {e}", file=sys.stderr)

        except (KeyboardInterrupt, EOFError):
            print("\n\033[33mExiting...\033[0m")
            break


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for ``chimol cli``."""
    parser = argparse.ArgumentParser(
        prog="chimol-cli",
        description="ChiMol Headless CLI — PyMOL-compatible molecular viewer commands",
    )
    parser.add_argument(
        "-e", "--execute",
        type=str,
        help="Execute a semicolon-separated list of commands and exit",
    )
    parser.add_argument(
        "-s", "--script",
        type=str,
        help="Run a script file (.pml) and exit",
    )
    parser.add_argument(
        "--no-ptpython",
        action="store_true",
        help="Force the basic input() REPL even if ptpython is installed",
    )
    args = parser.parse_args()

    cmd = _make_cmd()

    # ---- batch modes (no REPL) ----
    if args.execute:
        for line in args.execute.split(";"):
            cmd.do(line.strip())
        return

    if args.script:
        cmd.do(f"@{args.script}")
        return

    # ---- interactive REPL ----
    if args.no_ptpython:
        _run_fallback_repl(cmd)
        return

    try:
        import ptpython  # noqa: F401
        _run_ptpython_repl(cmd)
    except ImportError:
        _run_fallback_repl(cmd)


if __name__ == "__main__":
    main()
