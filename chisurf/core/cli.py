#!/usr/bin/env python

from __future__ import annotations

import ast
import importlib
import logging
import os
import pathlib
import pkgutil
import sys
from typing import Optional, Tuple, Dict, Iterable

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="chisurf", prog_name="csc")
def cli() -> None:
    """Command-line interface for chisurf tools and plugins.

    Use this command as a single entry point to run the CLIs provided by
    installed chisurf plugins. Each subcommand corresponds to one tool and
    comes with its own help and options.

    \b
    Examples:
      csc lltf --help
      csc burst-background --help
      csc count-rate analyze --help
    """

    # Ensure plugin-provided CLIs are registered the first time the
    # top-level CLI group is invoked, without doing any plugin discovery
    # at import time.
    _register_plugin_clis()


_LOG = logging.getLogger(__name__)

_PLUGINS_REGISTERED = False


def _parse_cli_entrypoint(entrypoint: str) -> Tuple[str, str, str]:
    """Parse strings of the form ``alias=module:attr``."""
    spec = (entrypoint or "").strip()
    if not spec or "=" not in spec:
        raise ValueError("Entry point must be in the form 'alias=module[:attr]'")
    alias, target = spec.split("=", 1)
    alias = alias.strip()
    if not alias:
        raise ValueError("Alias (command name) is empty")
    module_path, _, attr = target.partition(":")
    module_path = module_path.strip()
    attr = attr.strip() or "cli"
    if not module_path:
        raise ValueError("Module path is empty")
    return alias, module_path, attr


def _forward_plugin_cli(
    ctx: click.Context,
    *,
    module_path: str,
    attr_name: str,
    friendly_name: Optional[str] = None,
    command_name: Optional[str] = None,
) -> None:
    """Import the plugin CLI object and forward the current args."""
    try:
        module = importlib.import_module(module_path)
        target = getattr(module, attr_name)
    except Exception as exc:
        raise click.ClickException(
            f"Failed to import plugin CLI '{friendly_name or module_path}:{attr_name}': {exc}"
        )

    argv = list(ctx.args)
    if not argv:
        argv = ["--help"]

    runner = getattr(target, "main", None)
    if callable(runner):
        try:
            runner(args=argv, standalone_mode=True)
        except SystemExit as exc:  # pragma: no cover - normal Click exit path
            code = int(exc.code or 0)
            if code != 0:
                raise click.ClickException(
                    f"{friendly_name or module_path} exited with status {code}"
                )
        return

    # Fall back to calling the object directly if it behaves like a callable CLI.
    if callable(target):
        old_argv = list(sys.argv)
        try:
            # Simulate a classic ``if __name__ == '__main__'`` style CLI by
            # injecting our arguments into ``sys.argv``. This allows us to
            # wrap argparse-based entry points such as
            # ``chisurf.plugins.tttr.microtime_histogram.__main__.main``.
            prog = command_name or (friendly_name or module_path).split(".")[-1]
            sys.argv = [prog, *argv]
            target()  # type: ignore[call-arg]
        finally:
            sys.argv = old_argv
        return

    raise click.ClickException(
        f"Plugin CLI target '{module_path}:{attr_name}' is not callable"
    )


def _read_plugin_metadata(init_py: pathlib.Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not init_py.exists():
        return None, None, None
    try:
        source = init_py.read_text(encoding="utf-8")
    except Exception:
        return None, None, None
    try:
        tree = ast.parse(source, filename=str(init_py))
    except Exception:
        return None, None, None
    description = ast.get_docstring(tree) or "No description available."
    plugin_name = None
    cli_entrypoint = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in getattr(node, "targets", []):
            if isinstance(target, ast.Name) and target.id == "name":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    plugin_name = value.value
                elif isinstance(value, ast.Str):
                    plugin_name = value.s
            if isinstance(target, ast.Name) and target.id == "cli_entrypoint":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    cli_entrypoint = value.value.strip()
                elif isinstance(value, ast.Str):
                    cli_entrypoint = value.s.strip()
    return plugin_name, description, cli_entrypoint


def _discover_plugin_metadata() -> Iterable[Dict[str, object]]:
    """Yield metadata for built-in and user plugins **without importing packages**.

    This function performs a pure filesystem + AST scan over the built-in
    plugin tree (``chisurf/plugins``) and the user plugin directory
    (``~/.chisurf/plugins``). It never imports ``chisurf.plugins`` or any
    individual plugin modules, so CLI startup does not trigger any GUI
    initialization or heavy dependencies.
    """

    package_root = pathlib.Path(__file__).resolve().parent
    built_in_root = package_root / "plugins"
    user_root = pathlib.Path.home() / ".chisurf" / "plugins"

    search_roots = [
        ("built-in", built_in_root),
        ("user", user_root),
    ]

    seen = set()

    for source, root in search_roots:
        if not root.exists():
            continue
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root

        # Each __init__.py below the root corresponds to a candidate plugin
        # package under the chisurf.plugins.* namespace.
        for init_py in root_resolved.rglob("__init__.py"):
            try:
                rel_dir = init_py.parent.relative_to(root_resolved)
            except Exception:
                # Outside of the root we care about.
                continue

            # Skip the root package itself (e.g. chisurf.plugins).
            if str(rel_dir) == ".":
                continue

            parts = rel_dir.parts
            module_name = ".".join(parts)
            module_path = f"chisurf.plugins.{module_name}"

            key = (module_path, str(init_py.parent))
            if key in seen:
                continue

            plugin_name, description, cli_entrypoint = _read_plugin_metadata(init_py)

            # For CLI purposes we only care about packages that either
            # advertise a human-readable plugin name or explicitly opt into
            # the CLI via ``cli_entrypoint``. Plain namespace packages are
            # skipped entirely so they are never touched during CLI startup.
            if not plugin_name and not cli_entrypoint:
                continue

            seen.add(key)

            yield {
                "module_path": module_path,
                "module_name": parts[-1],
                "package_dir": init_py.parent,
                "source": source,
                "plugin_name": plugin_name,
                "description": description,
                "cli_entrypoint": cli_entrypoint,
            }


def _register_plugin_clis() -> None:
    """Discover plugin-provided CLI entry points and attach them to the main group.

    This function is idempotent and safe to call multiple times.
    """
    global _PLUGINS_REGISTERED

    if _PLUGINS_REGISTERED:
        return

    try:
        plugins = list(_discover_plugin_metadata())
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("Unable to discover plugins for CLI registration: %s", exc)
        return

    registered_specs = []  # for logging

    for metadata in plugins:
        entry_spec = metadata.get("cli_entrypoint")
        if not entry_spec:
            continue
        try:
            command_name, module_path, attr_name = _parse_cli_entrypoint(entry_spec)
        except ValueError as exc:
            _LOG.warning(
                "Skipping CLI entrypoint for plugin '%s': %s",
                metadata.get("module_path"),
                exc,
            )
            continue

        if command_name in cli.commands:
            _LOG.debug(
                "Skipping CLI entrypoint '%s' from plugin '%s' (command already defined)",
                command_name,
                metadata.get("module_path"),
            )
            continue

        # Derive a short help text from the plugin's docstring without
        # importing the plugin. We use the first non-empty line of the
        # description parsed from ``__init__.py``; if that is missing, fall
        # back to the human-readable plugin name or the module name.
        raw_desc = (metadata.get("description") or "").strip()
        if raw_desc:
            summary_line = raw_desc.splitlines()[0].strip()
        else:
            summary_line = ""

        help_text = summary_line or str(
            metadata.get("plugin_name") or metadata.get("module_name") or command_name
        )

        def _callback(ctx: click.Context, _command_name: str = command_name,
                      _module_path: str = module_path, _attr_name: str = attr_name,
                      _plugin_label: Optional[str] = metadata.get("plugin_name")) -> None:
            _forward_plugin_cli(
                ctx,
                module_path=_module_path,
                attr_name=_attr_name,
                friendly_name=_plugin_label or _module_path,
                command_name=_command_name,
            )

        command = click.Command(
            name=command_name,
            callback=click.pass_context(_callback),
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
            help=help_text,
        )
        cli.add_command(command)

        # Only record the public command alias for logging, not the full
        # module path, to keep the startup message compact.
        registered_specs.append(command_name)

    _PLUGINS_REGISTERED = True

    if registered_specs:
        _LOG.debug("Registered plugin CLIs: %s", ", ".join(registered_specs))


def _log_preloaded_plugin_modules() -> None:
    """Report any plugin modules that were already imported during CLI startup."""
    preloaded = sorted(
        name
        for name in sys.modules
        if name.startswith("chisurf.plugins.") and name.count(".") >= 2
    )
    if preloaded:
        _LOG.info(
            "Plugin modules already imported during chisurf.core.cli initialization: %s",
            ", ".join(preloaded),
        )


def main(argv: Optional[list[str]] = None) -> int:
    """Entry-point compatible wrapper.

    This allows calling chisurf.core.cli:main as well as chisurf.core.cli:cli.
    """

    if argv is None:
        argv = sys.argv[1:]

    # When invoked via python -m chisurf.core.cli or the installed console
    # script, register plugin CLIs and log any preloaded plugin modules
    # before delegating to Click's main dispatcher.
    _register_plugin_clis()
    _log_preloaded_plugin_modules()

    try:
        cli.main(args=argv, standalone_mode=True)
    except SystemExit as exc:  # normal click exit path
        return exc.code
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
