from __future__ import annotations

"""SV-05 guard: every published event topic must be declared in server_methods.json."""

import json
import pathlib
from importlib import resources
import ast
import inspect


def _published_topics() -> set[str]:
    """Return all event topics published via event_bus.publish(...) in server services."""
    services_dir = pathlib.Path(
        resources.files("chisurf.server.services").joinpath("__init__").__str__()
    ).parent
    topics: set[str] = set()
    for py_file in sorted(services_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        source = py_file.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "publish"
                    and node.args
                ):
                    topic = ast.literal_eval(node.args[0])
                    if isinstance(topic, str):
                        topics.add(topic)
    return topics


def _declared_topics() -> set[str]:
    """Return all event topics declared in server_methods.json."""
    with resources.files("chisurf.server").joinpath("server_methods.json").open() as fp:
        methods = json.load(fp)["methods"]
    topics: set[str] = set()
    for m in methods:
        for t in m.get("events", []):
            topics.add(t)
    return topics


def test_all_published_events_are_declared():
    """Every event_bus.publish(topic) must appear in some method's events array."""
    published = _published_topics()
    declared = _declared_topics()
    undeclared = published - declared
    assert not undeclared, (
        f"Published event topics not declared in server_methods.json: {sorted(undeclared)}"
    )


def test_all_declared_events_are_published():
    """Every event in server_methods.json must be published somewhere."""
    published = _published_topics()
    declared = _declared_topics()
    unpublished = declared - published
    assert not unpublished, (
        f"Declared event topics not published by any handler: {sorted(unpublished)}"
    )
