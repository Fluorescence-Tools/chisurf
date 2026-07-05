from __future__ import annotations

"""Contract tests for chisurf.server.services module structure."""

import json
from importlib import resources, import_module
from collections import Counter


def _service_modules() -> set[str]:
    """Return the set of service module names declared in server_methods.json."""
    with resources.files("chisurf.server").joinpath("server_methods.json").open() as fp:
        methods = json.load(fp)["methods"]
    modules: set[str] = set()
    for spec in methods:
        service = spec.get("service", "")
        if "." in service:
            modules.add(service.rsplit(".", 1)[0])
    return modules


def test_no_duplicate_handler_names():
    """BUG-02 guard: no service module defines the same top-level function twice."""
    for mod_name in sorted(_service_modules()):
        module = import_module(f"chisurf.server.services.{mod_name}")
        names = [n for n in dir(module) if not n.startswith("_")]
        dupes = {name for name, count in Counter(names).items() if count > 1}
        assert not dupes, (
            f"chisurf.server.services.{mod_name} has duplicate names: {dupes}"
        )
