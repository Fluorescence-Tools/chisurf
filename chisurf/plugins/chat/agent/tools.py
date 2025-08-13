import json
import os
import re
import pathlib
from typing import Any, Dict, Tuple, Callable, Optional, List

from dataclasses import dataclass

from ..tools.anisotropy import (
    run_anisotropy_analysis as _run_anisotropy,
    list_files as _list_files,
    read_text_head as _read_text_head,
    batch_run_anisotropy as _batch_run_anisotropy,
)

# Determine repo root from this file location: chisurf/plugins/chat/agent/tools.py -> parents[4]
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_WHITELIST = [
    PROJECT_ROOT,
    PROJECT_ROOT / 'test' / 'data',
]


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    func: Callable[..., Any]
    is_destructive: bool = False


def _norm_path(p: str) -> pathlib.Path:
    return pathlib.Path(p).expanduser().resolve()


def is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
        return str(path).startswith(str(root))
    except Exception:
        return False


def is_whitelisted_path(p: str, extra_whitelist: Optional[List[str]] = None) -> bool:
    path = _norm_path(p)
    roots = list(DEFAULT_WHITELIST)
    if extra_whitelist:
        for r in extra_whitelist:
            try:
                roots.append(_norm_path(r))
            except Exception:
                continue
    for r in roots:
        if is_within(path, r):
            return True
    return False


def _apply_defaults_and_validate(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    props = schema.get('properties', {})
    required = schema.get('required', [])
    out = dict(args or {})
    # apply defaults
    for key, prop in props.items():
        if key not in out and 'default' in prop:
            out[key] = prop['default']
    # check required
    for key in required:
        if key not in out:
            return out, f"Missing required argument: {key}"
    # type checks and constraints
    for key, prop in props.items():
        if key not in out:
            continue
        val = out[key]
        typ = prop.get('type')
        if typ == 'string':
            if not isinstance(val, str):
                return out, f"Argument '{key}' must be string"
        elif typ == 'number':
            if not isinstance(val, (int, float)):
                return out, f"Argument '{key}' must be number"
        elif typ == 'integer':
            if not isinstance(val, int):
                return out, f"Argument '{key}' must be integer"
            if 'minimum' in prop and val < int(prop['minimum']):
                return out, f"Argument '{key}' must be >= {prop['minimum']}"
        elif typ == 'boolean':
            if not isinstance(val, bool):
                return out, f"Argument '{key}' must be boolean"
        elif typ == 'object':
            if not isinstance(val, dict):
                return out, f"Argument '{key}' must be object"
        elif typ == 'array':
            if not isinstance(val, list):
                return out, f"Argument '{key}' must be array"
        # else: ignore unknown
    return out, None


def parse_toolcall_block(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    # Must be exactly fenced: ```toolcall ... ```
    # Support possible leading/trailing spaces before fences
    pattern = re.compile(r"```toolcall\s*\n([\s\S]*?)\n```", re.MULTILINE)
    m = pattern.search(text or '')
    if not m:
        return None
    block = m.group(1)
    try:
        data = json.loads(block)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    tool = data.get('tool')
    args = data.get('args')
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None
    return tool, args


# Tool Schemas
ANISOTROPY_SCHEMA = {
    "type": "object",
    "properties": {
        "csv_path":   {"type": "string", "description": "Absolute path to CSV"},
        "g_factor":   {"type": "number", "default": 1.0},
        "time_col":   {"type": "string", "default": "time"},
        "Ipar_col":   {"type": "string", "default": "I_par"},
        "Iperp_col":  {"type": "string", "default": "I_perp"},
        "smooth_window": {"type": "integer", "default": 1, "minimum": 1},
        "out_prefix": {"type": "string", "default": "anisotropy_out"},
    },
    "required": ["csv_path"],
}

LIST_FILES_SCHEMA = {
    "type": "object",
    "properties": {
        "dir_path": {"type": "string"},
        "pattern": {"type": "string", "default": "*.csv"},
        "max_items": {"type": "integer", "default": 200, "minimum": 1},
    },
    "required": ["dir_path"],
}

READ_TEXT_HEAD_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "n": {"type": "integer", "default": 2000, "minimum": 1},
    },
    "required": ["path"],
}

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "dir_path": {"type": "string"},
        "pattern": {"type": "string", "default": "*.csv"},
        "g_factor":   {"type": "number", "default": 1.0},
        "time_col":   {"type": "string", "default": "time"},
        "Ipar_col":   {"type": "string", "default": "I_par"},
        "Iperp_col":  {"type": "string", "default": "I_perp"},
        "smooth_window": {"type": "integer", "default": 1, "minimum": 1},
        "out_prefix": {"type": "string", "default": "anisotropy_out"},
    },
    "required": ["dir_path"],
}


_REGISTRY: Dict[str, ToolSpec] = {
    "run_anisotropy_analysis": ToolSpec(
        name="run_anisotropy_analysis",
        description="Compute time-resolved anisotropy from a CSV and save PNG/CSV.",
        schema=ANISOTROPY_SCHEMA,
        func=_run_anisotropy,
        is_destructive=False,
    ),
    "list_files": ToolSpec(
        name="list_files",
        description="List files matching a pattern within a directory (capped).",
        schema=LIST_FILES_SCHEMA,
        func=_list_files,
        is_destructive=False,
    ),
    "read_text_head": ToolSpec(
        name="read_text_head",
        description="Read first N characters of a text file (UTF-8).",
        schema=READ_TEXT_HEAD_SCHEMA,
        func=_read_text_head,
        is_destructive=False,
    ),
    "batch_run_anisotropy": ToolSpec(
        name="batch_run_anisotropy",
        description="Run anisotropy on all matching CSVs in a directory (destructive).",
        schema=BATCH_SCHEMA,
        func=_batch_run_anisotropy,
        is_destructive=True,
    ),
}


def list_tools() -> List[Dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "schema": t.schema, "is_destructive": t.is_destructive}
        for t in _REGISTRY.values()
    ]


def get_tool(name: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(name)


def validate_args(tool: ToolSpec, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    return _apply_defaults_and_validate(tool.schema, args)


def _enforce_whitelist(tool_name: str, args: Dict[str, Any], extra_whitelist: Optional[List[str]] = None) -> Optional[str]:
    # Check typical path args
    path_keys = [
        ('run_anisotropy_analysis', ['csv_path', 'out_prefix']),
        ('list_files', ['dir_path']),
        ('read_text_head', ['path']),
        ('batch_run_anisotropy', ['dir_path', 'out_prefix']),
    ]
    keys = []
    for tname, ks in path_keys:
        if tname == tool_name:
            keys = ks
            break
    for k in keys:
        v = args.get(k)
        if not isinstance(v, str):
            continue
        # Only enforce on absolute paths; for out_prefix we accept relative but join near inputs; server should enforce context
        if k in ("csv_path", "dir_path", "path"):
            if not pathlib.Path(v).is_absolute():
                return f"Argument '{k}' must be an absolute path"
            if not is_whitelisted_path(v, extra_whitelist=extra_whitelist):
                return f"Path not allowed by whitelist for '{k}'"
    return None


def execute_tool(tool_name: str, args: Dict[str, Any], confirm_destructive: bool = False, extra_whitelist: Optional[List[str]] = None) -> Tuple[Optional[Any], Optional[str]]:
    spec = get_tool(tool_name)
    if spec is None:
        return None, f"Unknown tool: {tool_name}"
    args2, err = validate_args(spec, args)
    if err:
        return None, err
    werr = _enforce_whitelist(tool_name, args2, extra_whitelist=extra_whitelist)
    if werr:
        return None, werr
    if spec.is_destructive and not confirm_destructive:
        return None, "Destructive tool requires confirmation. Resend with confirm_destructive=true."
    try:
        result = spec.func(**args2)
        return result, None
    except Exception as e:
        return None, str(e)
