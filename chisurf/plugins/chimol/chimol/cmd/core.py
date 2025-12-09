from __future__ import annotations

from pathlib import Path
from shlex import split as shlex_split
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
import tempfile
import urllib.error
import urllib.request
import re

import numpy as np
from ..analysis.metrics import compute_rmsd, compute_kabsch

if TYPE_CHECKING:
    from ..viewer.app.molview_main_window import MolViewPluginWindow


MessageCallback = Callable[[str], None]


class Cmd:
    def __init__(self, window: Optional["MolViewPluginWindow"] = None) -> None:
        self.window = window
        self._message_callback: Optional[MessageCallback] = None
        self._error_callback: Optional[MessageCallback] = None
        self._commands: Dict[str, Callable[[List[str]], object]] = {}
        self._named_selections: Dict[str, Dict[str, object]] = {}
        self._install_builtin_commands()

    def set_window(self, window: Optional["MolViewPluginWindow"]) -> None:
        self.window = window

    def set_message_callback(self, callback: Optional[MessageCallback]) -> None:
        self._message_callback = callback

    def set_error_callback(self, callback: Optional[MessageCallback]) -> None:
        self._error_callback = callback

    def register(self, name: str, func: Callable[[List[str]], object]) -> None:
        self._commands[name.lower()] = func

    def do(self, line: str) -> None:
        line = (line or "").strip()
        if not line:
            return

        # Script execution: '@filename' runs a text file with one command per line.
        if line.startswith("@"):
            script_path = line[1:].strip()
            if not script_path:
                self._emit_error("Usage: @<script_file>")
                return
            self._run_script_file(script_path)
            return

        try:
            parts = shlex_split(line)
        except Exception as exc:
            self._emit_error(f"Parse error: {exc}")
            return

        if not parts:
            return

        name = parts[0].lower()
        args = parts[1:]

        handler = self._commands.get(name)
        if handler is None:
            self._emit_error(
                f"Command '{name}' is not implemented in Moview/Chimol cmd (PyMOL compatibility layer)."
            )
            return

        try:
            result = handler(args)
        except Exception as exc:
            self._emit_error(f"Error in command '{name}': {exc}")
            return

        if result is not None:
            self._emit_message(str(result))

    def _run_script_file(self, path: str) -> None:
        """Execute a simple cmd script file, one command per non-empty line.

        Lines starting with '#' are treated as comments and skipped. This is
        similar in spirit to PyMOL's '@script.pml' support, but limited to the
        Moview cmd language (no arbitrary Python execution).
        """

        try:
            p = Path(path).expanduser()
        except Exception as exc:
            self._emit_error(f"Invalid script path {path!r}: {exc}")
            return

        if not p.exists():
            self._emit_error(f"Script file not found: {p}")
            return

        try:
            with p.open("rt", encoding="utf-8") as fh:
                for raw in fh:
                    line = (raw or "").strip()
                    if not line or line.startswith("#"):
                        continue
                    self.do(line)
        except Exception as exc:
            self._emit_error(f"Failed to run script {p!s}: {exc}")

    def _install_builtin_commands(self) -> None:
        self.register("help", self._cmd_help)
        self.register("load", self._cmd_load)
        self.register("open", self._cmd_load)
        self.register("fetch", self._cmd_fetch)
        self.register("fetch_emdb", self._cmd_fetch_emdb)
        self.register("fetch_ihm", self._cmd_fetch_ihm)
        self.register("bg_color", self._cmd_bg_color)
        self.register("bg_colour", self._cmd_bg_color)
        self.register("show", self._cmd_show)
        self.register("hide", self._cmd_hide)
        self.register("as", self._cmd_as)
        self.register("center", self._cmd_center)
        self.register("orient", self._cmd_orient)
        self.register("zoom", self._cmd_zoom)
        self.register("reset", self._cmd_reset)
        self.register("color", self._cmd_color)
        self.register("distance", self._cmd_distance)
        self.register("angle", self._cmd_angle)
        self.register("dihedral", self._cmd_dihedral)
        self.register("frame", self._cmd_frame)
        self.register("frame_next", self._cmd_frame_next)
        self.register("frame_prev", self._cmd_frame_prev)
        self.register("select", self._cmd_select)
        self.register("set", self._cmd_set)
        self.register("enable", self._cmd_enable)
        self.register("disable", self._cmd_disable)
        self.register("deselect", self._cmd_deselect)
        self.register("objects", self._cmd_objects)
        self.register("get_names", self._cmd_get_names)
        self.register("align", self._cmd_align)
        self.register("rms", self._cmd_rms)
        self.register("split_chains", self._cmd_split_chains)
        self.register("delete", self._cmd_delete)
        self.register("quit", self._cmd_quit)
        self.register("exit", self._cmd_quit)

    # ------------------------------------------------------------------
    # Python convenience API (thin wrappers around the internal commands)
    # ------------------------------------------------------------------

    def help(self) -> str:
        """Return a short help string listing available commands."""
        return self._cmd_help([])

    def load(self, *paths: str) -> None:
        """Load one or more structure files into the active viewer window."""
        self._cmd_load(list(paths))

    def open(self, *paths: str) -> None:
        """Alias for load()."""
        self._cmd_load(list(paths))

    def fetch(self, *pdb_ids: str) -> None:
        """Fetch one or more PDB IDs from RCSB and load them into the viewer.""" 
        self._cmd_fetch(list(pdb_ids))

    def fetch_emdb(self, *emdb_ids: str) -> None:
        self._cmd_fetch_emdb(list(emdb_ids))

    def fetch_ihm(self, *entry_ids: str) -> None:
        """Fetch one or more PDB-IHM CIF entries from pdb-ihm.org and load them."""

        self._cmd_fetch_ihm(list(entry_ids))

    def bg_color(self, color: str) -> None:
        """Set the viewer background color (PyMOL-style bg_color)."""
        self._cmd_bg_color([color])

    def show(self, rep: str) -> None:
        """Show a simple global representation (cartoon, sticks, surface, ...)."""
        self._cmd_show([rep])

    def hide(self, rep: str) -> None:
        """Hide a simple global representation (cartoon, sticks, surface, ...)."""
        self._cmd_hide([rep])

    def as_(self, rep: str) -> None:
        """Set the main representation mode (cartoon/lines/sticks/spheres)."""
        self._cmd_as([rep])

    def center(self) -> None:
        """Reset camera to show all visible objects (alias of orient/zoom/reset)."""
        self._cmd_center([])

    def orient(self) -> None:
        """Alias of center() for now."""
        self._cmd_orient([])

    def zoom(self) -> None:
        """Alias of center() for now."""
        self._cmd_zoom([])

    def reset(self) -> None:
        """Alias of center() for now."""
        self._cmd_reset([])

    def color(self, mode: str) -> None:
        """Set a simple color mode (single/by_residue/by_ss/by_sequence)."""
        self._cmd_color([mode])

    def distance(self, *tokens: str) -> None:
        """Measure distance between two selections (PyMOL-style).

        Syntax (minimal):

            distance sele1, sele2
            distance name, sele1, sele2

        where each ``seleX`` is a selection string such as::

            148l and res 50 and name ca
            res 10 and name nz
        """

        self._cmd_distance(list(tokens))

    def angle(self, *tokens: str) -> None:
        self._cmd_angle(list(tokens))

    def dihedral(self, *tokens: str) -> None:
        self._cmd_dihedral(list(tokens))

    def rms(self, *args: str) -> None:
        self._cmd_rms(list(args))

    def split_chains(self, prefix: Optional[str] = None) -> None:
        """Split objects into per-chain objects (PyMOL-style split_chains)."""
        args: list[str] = []
        if prefix:
            args.append(str(prefix))
        self._cmd_split_chains(args)

    def frame(self, index: int) -> None:
        self._cmd_frame([str(index)])
        self._cmd_frame_next([])

    def frame_prev(self) -> None:
        self._cmd_frame_prev([])

    def select(self, *tokens: str) -> None:
        """Select an object or object+residue (e.g. "1d3", "1dg3 and res 50")."""
        self._cmd_select(list(tokens))

    def set(self, name: str, value: str) -> None:
        """Minimal PyMOL-like set command (bg_color, color_mode, simple reps)."""
        self._cmd_set([name, value])

    def objects(self) -> None:
        """Print a summary of all loaded objects."""
        self._cmd_objects([])

    def get_names(self) -> None:
        """Print the list of object names (PyMOL-style get_names)."""
        self._cmd_get_names([])

    def enable(self, *tokens: str) -> None:
        """Enable object visibility (alias for object-based show)."""
        self._cmd_enable(list(tokens))

    def disable(self, *tokens: str) -> None:
        """Disable object visibility (alias for object-based hide)."""
        self._cmd_disable(list(tokens))

    def deselect(self) -> None:
        """Clear residue selection on the active object."""
        self._cmd_deselect([])

    def quit(self) -> None:
        """Close the attached ProtView window (if any)."""
        self._cmd_quit([])

    def exit(self) -> None:
        """Alias for quit()."""
        self._cmd_quit([])

    def _cmd_help(self, args: List[str]) -> str:
        names = sorted(self._commands.keys())
        return "Available commands: " + ", ".join(names)

    def _cmd_load(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: load <path> [more paths...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        for raw in args:
            path = Path(raw).expanduser()
            try:
                window._load_structure_from_path(path)
            except Exception as exc:
                self._emit_error(f"Failed to load '{path}': {exc}")
            else:
                self._emit_message(f"Loaded: {path}")

    def _cmd_fetch(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: fetch <pdb_id> [more ids...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        tmp_root = Path(tempfile.gettempdir())

        for raw in args:
            code = (raw or "").strip()
            if not code:
                continue
            pdb_id = code.lower()
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            dest = tmp_root / f"protview_{pdb_id}.pdb"

            try:
                with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
                    fh.write(resp.read())
            except Exception as exc:
                self._emit_error(f"Failed to fetch '{pdb_id}' from RCSB: {exc}")
                continue

            try:
                window._load_structure_from_path(dest, name=code)
            except Exception as exc:
                self._emit_error(f"Failed to load fetched PDB '{pdb_id}': {exc}")
            else:
                self._emit_message(f"Fetched and loaded PDB: {pdb_id}")

    def _cmd_fetch_emdb(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: fetch_emdb <emdb_id> [more ids...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        tmp_root = Path(tempfile.gettempdir())

        for raw in args:
            code = (raw or "").strip()
            if not code:
                continue
            m = re.search(r"(\d+)", code)
            if not m:
                self._emit_error(f"Could not parse EMDB id from {code!r}")
                continue
            emdb_num = m.group(1)
            folder = f"EMD-{emdb_num}"
            fname = f"emd_{emdb_num}.map.gz"
            url = (
                "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
                f"{folder}/map/{fname}"
            )
            dest = tmp_root / f"protview_emd_{emdb_num}.map.gz"

            try:
                with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
                    fh.write(resp.read())
            except Exception as exc:
                self._emit_error(f"Failed to fetch EMDB map '{code}': {exc}")
                continue

            try:
                window._load_structure_from_path(dest, name=f"EMD-{emdb_num}")
            except Exception as exc:
                self._emit_error(f"Failed to load EMDB map '{code}': {exc}")
            else:
                self._emit_message(f"Fetched and loaded EMDB map: EMD-{emdb_num}")

        
    def _cmd_fetch_ihm(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: fetch_ihm <entry_id> [more ids...]")
            return

        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return

        tmp_root = Path(tempfile.gettempdir())

        base_url = "https://pdb-ihm.org/cif"

        for raw in args:
            code = (raw or "").strip()
            if not code:
                continue
            entry_id = code.lower()
            url = f"{base_url}/{entry_id}.cif"
            dest = tmp_root / f"protview_ihm_{entry_id}.cif"

            try:
                with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
                    fh.write(resp.read())
            except Exception as exc:
                self._emit_error(
                    f"Failed to fetch IHM CIF '{entry_id}' from pdb-ihm.org: {exc}"
                )
                continue

            try:
                window._load_structure_from_path(dest, name=code)
            except Exception as exc:
                self._emit_error(f"Failed to load fetched IHM CIF '{entry_id}': {exc}")
            else:
                self._emit_message(f"Fetched and loaded IHM CIF: {entry_id}")

    # ------------------------------------------------------------------
    # Tier 1 viewer helpers
    # ------------------------------------------------------------------

    def _require_window_and_viewer(self):
        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return None, None
        viewer = getattr(window, "viewer", None)
        if viewer is None:
            self._emit_error("Attached window has no 'viewer' attribute")
            return window, None
        return window, viewer

    def _cmd_bg_color(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: bg_color <color>")
            return
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return
        color = args[0]
        try:
            viewer.set_background_color(color)
        except Exception as exc:
            self._emit_error(f"Failed to set background color: {exc}")

    def _cmd_show(self, args: List[str]) -> None:
        self._cmd_toggle_representation(args, visible=True)

    def _cmd_hide(self, args: List[str]) -> None:
        self._cmd_toggle_representation(args, visible=False)

    def _cmd_as(self, args: List[str]) -> None:
        """Set the primary representation mode (cartoon/lines/sticks/spheres)."""

        if not args:
            self._emit_error("Usage: as <cartoon|lines|sticks|spheres>")
            return

        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        rep = (args[0] or "").strip().lower()
        if rep in ("cartoon", "ribbon"):
            try:
                viewer.set_representation("cartoon")
            except Exception as exc:
                self._emit_error(f"Failed to set representation: {exc}")
            return
        if rep in ("lines", "trace", "ca_trace"):
            try:
                viewer.set_representation("ca_trace")
            except Exception as exc:
                self._emit_error(f"Failed to set representation: {exc}")
            return
        if rep in ("spheres", "atoms", "balls", "ball"):
            try:
                viewer.set_representation("atoms")
            except Exception as exc:
                self._emit_error(f"Failed to set representation: {exc}")
            return

        self._emit_error(f"Unsupported representation for 'as': {rep}")

    def _cmd_toggle_representation(self, args: List[str], *, visible: bool) -> None:
        if not args:
            self._emit_error("Usage: show/hide <cartoon|trace|atoms|sticks|dots|surface|plane>")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        rep = (args[0] or "").strip().lower()
        vis = bool(visible)

        if rep in ("all", "*"):
            try:
                objects = viewer.list_objects()
            except Exception as exc:
                self._emit_error(f"Failed to list objects: {exc}")
                return
            for obj in objects:
                oid = obj.get("id")
                if not oid:
                    continue
                try:
                    if hasattr(window, "_set_object_visible"):
                        window._set_object_visible(str(oid), vis)
                    else:
                        viewer.set_object_visible(str(oid), vis)
                except Exception:
                    continue
            return

        try:
            if rep in ("cartoon", "ribbon"):
                viewer.set_cartoon_visible(vis)
            elif rep in ("trace", "ca_trace", "lines"):
                viewer.set_trace_visible(vis)
            elif rep in ("atoms", "spheres", "balls", "ball"):
                viewer.set_atoms_visible_all(vis)
            elif rep in ("sticks", "bonds"):
                viewer.set_sticks_visible(vis)
            elif rep in ("dots", "points"):
                viewer.set_dots_visible(vis)
            elif rep in ("surface", "surf"):
                viewer.set_surface_visible(vis)
            elif rep in ("plane", "grid"):
                viewer.set_plane_visible(vis)
            else:
                self._emit_error(
                    f"Unsupported representation for show/hide: {rep}"
                )
                return
        except Exception as exc:
            self._emit_error(f"Failed to update representation '{rep}': {exc}")

    def _cmd_center(self, args: List[str]) -> None:
        """Center view on all visible objects (currently same as reset)."""

        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return
        try:
            viewer.reset_view()
        except Exception as exc:
            self._emit_error(f"Failed to center view: {exc}")

    def _cmd_orient(self, args: List[str]) -> None:
        """Orient view (alias of center/reset for now)."""

        self._cmd_center(args)

    def _cmd_zoom(self, args: List[str]) -> None:
        """Zoom view (alias of center/reset for now)."""

        self._cmd_center(args)

    def _cmd_reset(self, args: List[str]) -> None:
        """Reset view (alias of center for now)."""

        self._cmd_center(args)

    def _cmd_color(self, args: List[str]) -> None:
        """Set a simple color mode (single/by_residue/by_ss/by_sequence)."""

        if not args:
            self._emit_error(
                "Usage: color <single|by_residue|by_ss|by_sequence>[, selection] "
                "or color <color>, selection"
            )
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        joined = " ".join(args).strip()

        # Comma-separated form: either mode + selection or color + selection.
        if "," in joined:
            parts = [part.strip() for part in joined.split(",", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                self._emit_error(
                    "Usage: color <mode>, selection or color <color>, selection"
                )
                return
            left, right = parts

            raw_left = left.lower()

            # Legacy/Mode form: color <mode>, selection
            try:
                mode = self._normalize_color_mode(raw_left)
            except ValueError:
                mode = None

            if mode is not None:
                try:
                    self._apply_color_mode(viewer, mode, selection=right)
                except ValueError as exc:
                    self._emit_error(str(exc))
                else:
                    try:
                        if window is not None:
                            window._update_sequence_view()
                    except Exception:
                        pass
                return

            # PyMOL-style coloring: color <color>, selection
            color_spec_first = left
            sele_expr_first = right

            rgba = None
            sele_expr = None
            try:
                rgba = self._parse_color_spec(color_spec_first)
                sele_expr = sele_expr_first
            except ValueError:
                # Fallback for the user's original order: color selection, color
                try:
                    rgba = self._parse_color_spec(sele_expr_first)
                    sele_expr = color_spec_first
                except ValueError:
                    self._emit_error(
                        "Usage: color <single|by_residue|by_ss|by_sequence>[, selection] "
                        "or color <color>, selection"
                    )
                    return

            if rgba is None or sele_expr is None:
                self._emit_error("Could not parse color/selection for color command")
                return

            try:
                self._apply_color_to_selection(viewer, sele_expr, rgba)
            except ValueError as exc:
                self._emit_error(str(exc))
            else:
                try:
                    if window is not None:
                        window._update_sequence_view()
                except Exception:
                    pass
            return

        # No comma: treat as simple color-mode toggle on active object.
        raw = (args[0] or "").strip().lower()
        try:
            mode = self._normalize_color_mode(raw)
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            self._apply_color_mode(viewer, mode, selection=None)
        except ValueError as exc:
            self._emit_error(str(exc))
        else:
            try:
                if window is not None:
                    window._update_sequence_view()
            except Exception:
                pass
    def _normalize_color_mode(self, raw: str) -> str:
        token = (raw or "").strip().lower()
        if token in ("single", "uniform"):
            return "single"
        if token in ("by_residue", "residue", "aa", "by_aa"):
            return "by_residue"
        if token in (
            "by_ss",
            "ss",
            "secondary",
            "by_secondary_structure",
        ):
            return "by_secondary_structure"
        if token in ("by_sequence", "sequence", "seq"):
            return "by_sequence"
        raise ValueError(
            "Unsupported color mode. Use one of: "
            "single, by_residue, by_ss, by_sequence."
        )

        
    def _apply_color_mode(self, viewer, mode: str, *, selection: Optional[str]) -> None:
        # Clear any explicit overrides so the mode is visible.
        clear_overrides = getattr(viewer, "clear_color_overrides", None)

        if not selection:
            if callable(clear_overrides):
                try:
                    clear_overrides()
                except Exception:
                    pass
            viewer.set_color_mode(mode)
            self._emit_message(f"Color mode set to {mode}")
            return

        obj_id, obj_name, _ = self._resolve_selection_to_residue_indices(
            viewer, selection
        )
        if not obj_id:
            raise ValueError("Selection did not resolve to an object")

        activate = getattr(viewer, "_activate_object", None)
        if callable(activate):
            try:
                with activate(obj_id):
                    if callable(clear_overrides):
                        try:
                            clear_overrides()
                        except Exception:
                            pass
                    viewer.set_color_mode(mode)
            except Exception as exc:
                raise ValueError(f"Failed to set color mode on {obj_name}: {exc}")
        else:
            try:
                viewer.set_active_object(obj_id)
                if callable(clear_overrides):
                    try:
                        clear_overrides()
                    except Exception:
                        pass
                viewer.set_color_mode(mode)
            except Exception as exc:
                raise ValueError(f"Failed to set color mode on {obj_name}: {exc}")

        self._emit_message(f"Color mode for {obj_name} set to {mode}")

    def _parse_color_spec(self, spec: str) -> np.ndarray:
        text = (spec or "").strip()
        if not text:
            raise ValueError("Empty color specification")

        name = text.lower()
        named: Dict[str, tuple[float, float, float]] = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "gray": (0.5, 0.5, 0.5),
            "grey": (0.5, 0.5, 0.5),
            "orange": (1.0, 0.5, 0.0),
        }

        if name in named:
            r, g, b = named[name]
            return np.array([r, g, b, 1.0], dtype=float)

        if name.startswith("#") and len(name) in (7, 9):
            try:
                r = int(name[1:3], 16) / 255.0
                g = int(name[3:5], 16) / 255.0
                b = int(name[5:7], 16) / 255.0
                a = (
                    int(name[7:9], 16) / 255.0
                    if len(name) == 9
                    else 1.0
                )
            except Exception:
                raise ValueError(f"Invalid hex color: {spec!r}")
            return np.array([r, g, b, a], dtype=float)

        # Fallback: try comma- or space-separated numeric triplet/quadruplet.
        for sep in (",", " "):
            if sep in text:
                parts = [p for p in text.replace(",", " ").split() if p]
                if not parts:
                    break
                vals: List[float] = []
                for p in parts:
                    try:
                        v = float(p)
                    except Exception:
                        raise ValueError(f"Invalid color component {p!r} in {spec!r}")
                    if v > 1.0:
                        v = v / 255.0
                    vals.append(v)
                if len(vals) == 3:
                    vals.append(1.0)
                if len(vals) != 4:
                    raise ValueError(f"Color spec {spec!r} must have 3 or 4 components")
                return np.asarray(vals, dtype=float)

        raise ValueError(f"Unrecognized color specification: {spec!r}")

    def _apply_color_to_selection(
        self,
        viewer,
        sele_expr: str,
        rgba: np.ndarray,
    ) -> None:
        obj_id, obj_name, res_indices = self._resolve_selection_to_residue_indices(
            viewer, sele_expr
        )
        if not obj_id:
            raise ValueError("Selection did not resolve to an object")

        try:
            entry = viewer._objects.get(obj_id)  # type: ignore[attr-defined]
        except Exception:
            entry = None
        if entry is None:
            raise ValueError(f"Unknown object in selection: {obj_name}")

        state = getattr(entry, "state", None)
        if state is None:
            raise ValueError(f"Object {obj_name} has no state")

        all_atom_res_ids = getattr(state, "all_atom_res_ids", None)
        residue_ids = getattr(state, "residue_ids", None)
        all_atom_coords = getattr(state, "all_atom_coords", None)

        if (
            all_atom_res_ids is None
            or residue_ids is None
            or all_atom_coords is None
        ):
            raise ValueError(
                f"Object {obj_name} does not expose atom-level coordinates for coloring"
            )

        try:
            atom_res_ids_arr = np.asarray(all_atom_res_ids)
            res_ids_arr = np.asarray(residue_ids)
        except Exception:
            raise ValueError(f"Could not access atom/residue ids for object {obj_name}")

        try:
            n_atoms = int(np.asarray(all_atom_coords).shape[0])
        except Exception:
            raise ValueError(f"Invalid atom coordinate array on object {obj_name}")

        # Build/extend per-atom override array with NaN -> no override.
        try:
            cur_atom = np.asarray(state.colors_per_atom_override, dtype=float)
        except Exception:
            cur_atom = None
        if cur_atom is None or cur_atom.ndim != 2 or cur_atom.shape[0] != n_atoms:
            cur_atom = np.full((n_atoms, 4), np.nan, dtype=float)

        rgba4 = np.asarray(rgba, dtype=float).reshape(4)

        for ri in res_indices:
            if ri < 0 or ri >= res_ids_arr.shape[0]:
                continue
            rid = res_ids_arr[ri]
            try:
                mask = atom_res_ids_arr == rid
            except Exception:
                continue
            if not np.any(mask):
                continue
            cur_atom[mask, :] = rgba4

        state.colors_per_atom_override = cur_atom

        # Also update per-residue override array so CA-based colors match.
        try:
            n_res = int(res_ids_arr.shape[0])
        except Exception:
            n_res = 0
        if n_res > 0:
            try:
                cur_res = np.asarray(state.colors_per_residue_override, dtype=float)
            except Exception:
                cur_res = None
            if cur_res is None or cur_res.ndim != 2 or cur_res.shape[0] != n_res:
                cur_res = np.full((n_res, 4), np.nan, dtype=float)
            for ri in res_indices:
                if 0 <= ri < n_res:
                    cur_res[ri, :] = rgba4
            state.colors_per_residue_override = cur_res

        try:
            viewer._update_view()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _cmd_distance(self, args: List[str]) -> None:
        """Measure distance between two selections (PyMOL-style)."""

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: distance sele1, sele2")
            return

        try:
            meas_name, sele_parts = self._parse_measurement_selections(
                args, expected_count=2, cmd="distance"
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        sele1, sele2 = sele_parts

        try:
            obj1_id, obj1_name, res_i, atom1, p1 = self._resolve_selection_to_atom(
                viewer, sele1
            )
            obj2_id, obj2_name, res_j, atom2, p2 = self._resolve_selection_to_atom(
                viewer, sele2
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            v1 = np.asarray(p1, dtype=float).reshape(-1)
            v2 = np.asarray(p2, dtype=float).reshape(-1)
        except Exception:
            self._emit_error("Could not compute distance (no coordinates)")
            return

        if v1.shape[0] != 3 or v2.shape[0] != 3:
            self._emit_error("Could not compute distance (invalid coordinates)")
            return

        try:
            dist = float(np.linalg.norm(v1 - v2))
        except Exception:
            self._emit_error("Could not compute distance (no coordinates)")
            return

        label1 = f"{obj1_name} res {res_i + 1}"
        if atom1:
            label1 += f" and name {atom1}"
        label2 = f"{obj2_name} res {res_j + 1}"
        if atom2:
            label2 += f" and name {atom2}"

        prefix = "distance "
        if meas_name:
            prefix += f"{meas_name} "

        self._emit_message(f"{prefix}{label1} - {label2}: {dist:.3f}")

    def _cmd_frame(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: frame [object] index")
            return

        tokens = list(args)
        obj_id: Optional[str] = None
        obj_name: Optional[str] = None

        if len(tokens) == 1:
            try:
                active_id = viewer.get_active_object_id()
            except Exception:
                active_id = None
            if active_id is None:
                self._emit_error("No active object for frame control")
                return
            obj_id = str(active_id)
            obj_info = self._find_object_by_name(viewer, obj_id)
            obj_name = str(obj_info.get("name")) if obj_info is not None else obj_id
            idx_str = tokens[0]
        else:
            obj_token = tokens[0]
            obj_info = self._find_object_by_name(viewer, obj_token)
            if obj_info is None:
                self._emit_error(f"Unknown object: {obj_token}")
                return
            obj_id = str(obj_info.get("id"))
            obj_name = str(obj_info.get("name"))
            idx_str = tokens[1]

        try:
            frame_no = int(idx_str)
        except Exception:
            self._emit_error("Frame index must be an integer")
            return

        if frame_no <= 0:
            frame_idx = 0
        else:
            frame_idx = frame_no - 1

        try:
            viewer.set_active_frame(frame_idx, object_id=obj_id)
        except Exception as exc:
            self._emit_error(f"Failed to set frame on {obj_name}: {exc}")
            return

        try:
            n_frames = viewer.get_frame_count(obj_id)
        except Exception:
            n_frames = 0

        if n_frames > 0:
            if frame_no > n_frames:
                frame_no = n_frames
            self._emit_message(
                f"Frame for {obj_name}: {frame_no}/{n_frames}"
            )
        else:
            self._emit_message(f"Frame for {obj_name}: {frame_no}")

    def _cmd_frame_next(self, args: List[str]) -> None:
        self._cmd_frame_step(1)

    def _cmd_frame_prev(self, args: List[str]) -> None:
        self._cmd_frame_step(-1)

    def _cmd_frame_step(self, delta: int) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        try:
            obj_id = viewer.get_active_object_id()
        except Exception:
            obj_id = None
        if obj_id is None:
            self._emit_error("No active object for frame control")
            return

        try:
            n_frames = viewer.get_frame_count(obj_id)
        except Exception:
            n_frames = 0
        if n_frames <= 0:
            self._emit_error("Active object has no frames")
            return

        try:
            current = viewer.get_active_frame_index(obj_id)
        except Exception:
            current = 0

        idx = current + int(delta)
        if idx < 0:
            idx = 0
        if idx >= n_frames:
            idx = n_frames - 1

        try:
            viewer.set_active_frame(idx, object_id=obj_id)
        except Exception:
            return

        self._emit_message(f"Frame: {idx + 1}/{n_frames}")

    def _compute_residue_distance(self, viewer, object_id: str, i: int, j: int) -> Optional[float]:
        """Return Euclidean distance between two residue indices for an object.

        This uses the internal coords array for the object state. It assumes
        that residue indices used in selections correspond to rows in that
        coordinate array (as is the case for set_selected_residues).
        """

        try:
            entry = viewer._objects.get(object_id)  # type: ignore[attr-defined]
        except Exception:
            return None
        if entry is None:
            return None

        state = getattr(entry, "state", None)
        coords = getattr(state, "coords", None)
        if coords is None:
            return None

        try:
            arr = np.asarray(coords, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None

        n = arr.shape[0]
        if i < 0 or j < 0 or i >= n or j >= n:
            return None

        try:
            p_i = arr[i]
            p_j = arr[j]
            return float(np.linalg.norm(p_i - p_j))
        except Exception:
            return None

    def _compute_residue_distance_two(
        self,
        viewer,
        object_id_1: str,
        i: int,
        object_id_2: str,
        j: int,
    ) -> Optional[float]:
        try:
            entry1 = viewer._objects.get(object_id_1)  # type: ignore[attr-defined]
            entry2 = viewer._objects.get(object_id_2)  # type: ignore[attr-defined]
        except Exception:
            return None
        if entry1 is None or entry2 is None:
            return None

        state1 = getattr(entry1, "state", None)
        state2 = getattr(entry2, "state", None)
        coords1 = getattr(state1, "coords", None)
        coords2 = getattr(state2, "coords", None)
        if coords1 is None or coords2 is None:
            return None

        try:
            arr1 = np.asarray(coords1, dtype=float)
            arr2 = np.asarray(coords2, dtype=float)
        except Exception:
            return None
        if arr1.ndim != 2 or arr1.shape[1] != 3:
            return None
        if arr2.ndim != 2 or arr2.shape[1] != 3:
            return None

        n1 = arr1.shape[0]
        n2 = arr2.shape[0]
        if i < 0 or j < 0 or i >= n1 or j >= n2:
            return None

        try:
            p_i = arr1[i]
            p_j = arr2[j]
            return float(np.linalg.norm(p_i - p_j))
        except Exception:
            return None

    def _cmd_angle(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: angle sele1, sele2, sele3")
            return

        try:
            meas_name, sele_parts = self._parse_measurement_selections(
                args, expected_count=3, cmd="angle"
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            obj1_id, obj1_name, res_i, atom1, p1 = self._resolve_selection_to_atom(
                viewer, sele_parts[0]
            )
            obj2_id, obj2_name, res_j, atom2, p2 = self._resolve_selection_to_atom(
                viewer, sele_parts[1]
            )
            obj3_id, obj3_name, res_k, atom3, p3 = self._resolve_selection_to_atom(
                viewer, sele_parts[2]
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            v1 = np.asarray(p1, dtype=float).reshape(-1)
            v2 = np.asarray(p2, dtype=float).reshape(-1)
            v3 = np.asarray(p3, dtype=float).reshape(-1)
        except Exception:
            self._emit_error("Could not compute angle (no coordinates)")
            return

        if v1.shape[0] != 3 or v2.shape[0] != 3 or v3.shape[0] != 3:
            self._emit_error("Could not compute angle (invalid coordinates)")
            return

        try:
            a = v1 - v2
            b = v3 - v2
            n1 = float(np.linalg.norm(a))
            n2 = float(np.linalg.norm(b))
            if n1 <= 0.0 or n2 <= 0.0:
                self._emit_error("Could not compute angle (degenerate geometry)")
                return
            cos_theta = float(np.dot(a, b) / (n1 * n2))
            if cos_theta > 1.0:
                cos_theta = 1.0
            if cos_theta < -1.0:
                cos_theta = -1.0
            value = float(np.degrees(np.arccos(cos_theta)))
        except Exception:
            self._emit_error("Could not compute angle (no coordinates)")
            return

        label1 = f"{obj1_name} res {res_i + 1}"
        if atom1:
            label1 += f" and name {atom1}"
        label2 = f"{obj2_name} res {res_j + 1}"
        if atom2:
            label2 += f" and name {atom2}"
        label3 = f"{obj3_name} res {res_k + 1}"
        if atom3:
            label3 += f" and name {atom3}"

        prefix = "angle "
        if meas_name:
            prefix += f"{meas_name} "

        self._emit_message(
            f"{prefix}{label1} - {label2} - {label3}: {value:.3f}"
        )

    def _selection_requests_ca_only(self, expr: str) -> bool:
        text = (expr or "").strip()
        if not text:
            return False

        try:
            tokens = shlex_split(text)
        except Exception:
            return False

        found_ca = False
        other_named = False
        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i].lower()
            if tok == "name" and i + 1 < n:
                name_tok = tokens[i + 1].strip().lower()
                if name_tok == "ca":
                    found_ca = True
                else:
                    other_named = True
                i += 2
                continue
            i += 1

        return found_ca and not other_named

    def _gather_atom_coords_for_residues(
        self,
        viewer,
        obj_id: str,
        obj_name: str,
        res_indices: Optional[List[int]],
    ) -> np.ndarray:
        """Return per-atom coordinates for the given residue indices.

        If ``res_indices`` is ``None`` or empty, all atoms on the object are
        returned. Coordinates are ordered as in ``all_atom_coords`` and filtered
        by matching ``all_atom_res_ids`` against residue ids.
        """

        try:
            entry = viewer._objects.get(obj_id)  # type: ignore[attr-defined]
        except Exception:
            entry = None
        if entry is None:
            raise ValueError(f"Unknown object in selection: {obj_name}")

        state = getattr(entry, "state", None)
        all_atom_coords = getattr(state, "all_atom_coords", None)
        all_atom_res_ids = getattr(state, "all_atom_res_ids", None)
        residue_ids = getattr(state, "residue_ids", None)

        if all_atom_coords is None or all_atom_res_ids is None or residue_ids is None:
            raise ValueError(
                f"Object {obj_name} does not expose atom-level coordinates for RMSD "
                "(try using '... and name ca' to fall back to CA-based RMSD)."
            )

        try:
            atom_coords_arr = np.asarray(all_atom_coords, dtype=float)
            atom_res_ids_arr = np.asarray(all_atom_res_ids)
            res_ids_arr = np.asarray(residue_ids)
        except Exception:
            raise ValueError(f"Could not access atom/residue ids for object {obj_name}")

        if atom_coords_arr.ndim != 2 or atom_coords_arr.shape[1] != 3:
            raise ValueError(f"Invalid atom coordinate array on object {obj_name}")

        # If no residue indices were given, use all atoms.
        if not res_indices:
            return atom_coords_arr.copy()

        keep_mask = np.zeros(atom_coords_arr.shape[0], dtype=bool)
        for ri in res_indices:
            if ri < 0 or ri >= res_ids_arr.shape[0]:
                continue
            rid = res_ids_arr[ri]
            try:
                keep_mask |= atom_res_ids_arr == rid
            except Exception:
                continue

        if not np.any(keep_mask):
            return np.zeros((0, 3), dtype=float)

        return atom_coords_arr[keep_mask].copy()

    def _cmd_rms(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: rms mobile_selection, target_selection")
            return

        try:
            _, parts = self._parse_measurement_selections(
                args, expected_count=2, cmd="rms"
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        mobile_expr, target_expr = parts

        ca_mobile = self._selection_requests_ca_only(mobile_expr)
        ca_target = self._selection_requests_ca_only(target_expr)
        use_ca = ca_mobile and ca_target

        try:
            mob_obj, mob_name, mob_indices = self._resolve_selection_to_residue_indices(
                viewer, mobile_expr
            )
            tgt_obj, tgt_name, tgt_indices = self._resolve_selection_to_residue_indices(
                viewer, target_expr
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            if use_ca:
                mob_coords = viewer.get_residue_positions(
                    mob_indices if mob_indices else None, object_id=mob_obj
                )
                tgt_coords = viewer.get_residue_positions(
                    tgt_indices if tgt_indices else None, object_id=tgt_obj
                )
            else:
                mob_coords = self._gather_atom_coords_for_residues(
                    viewer,
                    mob_obj,
                    mob_name,
                    mob_indices if mob_indices else None,
                )
                tgt_coords = self._gather_atom_coords_for_residues(
                    viewer,
                    tgt_obj,
                    tgt_name,
                    tgt_indices if tgt_indices else None,
                )
        except Exception as exc:
            self._emit_error(f"Failed to access coordinates for RMSD: {exc}")
            return

        if mob_coords.size == 0 or tgt_coords.size == 0:
            self._emit_error("Selections must contain at least one coordinate")
            return

        count = min(mob_coords.shape[0], tgt_coords.shape[0])
        if count <= 0:
            self._emit_error("Selections did not yield matching coordinate counts")
            return

        mob_coords = mob_coords[:count]
        tgt_coords = tgt_coords[:count]

        try:
            rmsd = compute_rmsd(mob_coords, tgt_coords)
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        if use_ca:
            msg = (
                f"RMSD between {mob_name} and {tgt_name} over {count} CA atoms: "
                f"{rmsd:.3f} Å"
            )
        else:
            msg = (
                f"RMSD between {mob_name} and {tgt_name} over {count} atoms: "
                f"{rmsd:.3f} Å"
            )
        self._emit_message(msg)

    def _cmd_split_chains(self, args: List[str]) -> None:
        """Split objects into per-chain objects (PyMOL-style split_chains)."""

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        prefix = args[0] if args else None
        try:
            count = viewer.split_chains(prefix=prefix)
            try:
                window._refresh_objects_from_viewer()
            except Exception:
                pass
        except Exception as exc:
            self._emit_error(f"Failed to split chains: {exc}")
            return

        self._emit_message(f"Created {count} chain object(s)")

    def _cmd_align(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: align mobile_selection, target_selection")
            return

        try:
            _, parts = self._parse_measurement_selections(
                args, expected_count=2, cmd="align"
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        mobile_expr, target_expr = parts

        try:
            mob_obj, mob_name, mob_indices = self._resolve_selection_to_residue_indices(
                viewer, mobile_expr
            )
            tgt_obj, tgt_name, tgt_indices = self._resolve_selection_to_residue_indices(
                viewer, target_expr
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            mob_coords = viewer.get_residue_positions(
                mob_indices if mob_indices else None, object_id=mob_obj
            )
            tgt_coords = viewer.get_residue_positions(
                tgt_indices if tgt_indices else None, object_id=tgt_obj
            )
        except Exception as exc:
            self._emit_error(f"Failed to access coordinates for alignment: {exc}")
            return

        if mob_coords.size == 0 or tgt_coords.size == 0:
            self._emit_error("Selections must contain at least one residue")
            return

        count = min(mob_coords.shape[0], tgt_coords.shape[0])
        if count < 3:
            self._emit_error("Alignment requires at least three residues in each selection")
            return

        mob_coords = mob_coords[:count]
        tgt_coords = tgt_coords[:count]

        try:
            rot, trans, rmsd = compute_kabsch(mob_coords, tgt_coords)
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            viewer.apply_transform_to_object(rot.T, trans, object_id=mob_obj)
        except Exception as exc:
            self._emit_error(f"Failed to apply alignment transform: {exc}")
            return

        self._emit_message(
            f"Aligned {mob_name} onto {tgt_name} using {count} residues "
            f"(fit RMSD over selection: {rmsd:.3f} Å)"
        )


    def _cmd_dihedral(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error("Usage: dihedral sele1, sele2, sele3, sele4")
            return

        try:
            meas_name, sele_parts = self._parse_measurement_selections(
                args, expected_count=4, cmd="dihedral"
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            obj1_id, obj1_name, res_i, atom1, p1 = self._resolve_selection_to_atom(
                viewer, sele_parts[0]
            )
            obj2_id, obj2_name, res_j, atom2, p2 = self._resolve_selection_to_atom(
                viewer, sele_parts[1]
            )
            obj3_id, obj3_name, res_k, atom3, p3 = self._resolve_selection_to_atom(
                viewer, sele_parts[2]
            )
            obj4_id, obj4_name, res_l, atom4, p4 = self._resolve_selection_to_atom(
                viewer, sele_parts[3]
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        try:
            v1 = np.asarray(p1, dtype=float).reshape(-1)
            v2 = np.asarray(p2, dtype=float).reshape(-1)
            v3 = np.asarray(p3, dtype=float).reshape(-1)
            v4 = np.asarray(p4, dtype=float).reshape(-1)
        except Exception:
            self._emit_error("Could not compute dihedral (no coordinates)")
            return

        if (
            v1.shape[0] != 3
            or v2.shape[0] != 3
            or v3.shape[0] != 3
            or v4.shape[0] != 3
        ):
            self._emit_error("Could not compute dihedral (invalid coordinates)")
            return

        try:
            b0 = v2 - v1
            b1 = v3 - v2
            b2 = v4 - v3

            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)
            if np.linalg.norm(n1) <= 0.0 or np.linalg.norm(n2) <= 0.0:
                self._emit_error("Could not compute dihedral (degenerate geometry)")
                return

            n1_u = n1 / np.linalg.norm(n1)
            n2_u = n2 / np.linalg.norm(n2)
            b1_u = b1 / np.linalg.norm(b1) if np.linalg.norm(b1) > 0.0 else b1

            m1 = np.cross(n1_u, b1_u)
            x = float(np.dot(n1_u, n2_u))
            y = float(np.dot(m1, n2_u))
            value = float(np.degrees(np.arctan2(y, x)))
        except Exception:
            self._emit_error("Could not compute dihedral (no coordinates)")
            return

        label1 = f"{obj1_name} res {res_i + 1}"
        if atom1:
            label1 += f" and name {atom1}"
        label2 = f"{obj2_name} res {res_j + 1}"
        if atom2:
            label2 += f" and name {atom2}"
        label3 = f"{obj3_name} res {res_k + 1}"
        if atom3:
            label3 += f" and name {atom3}"
        label4 = f"{obj4_name} res {res_l + 1}"
        if atom4:
            label4 += f" and name {atom4}"

        prefix = "dihedral "
        if meas_name:
            prefix += f"{meas_name} "

        self._emit_message(
            prefix + f"{label1} - {label2} - {label3} - {label4}: {value:.3f}"
        )

    def _compute_residue_angle(
        self,
        viewer,
        object_id: str,
        i: int,
        j: int,
        k: int,
    ) -> Optional[float]:
        try:
            entry = viewer._objects.get(object_id)  # type: ignore[attr-defined]
        except Exception:
            return None
        if entry is None:
            return None

        state = getattr(entry, "state", None)
        coords = getattr(state, "coords", None)
        if coords is None:
            return None

        try:
            arr = np.asarray(coords, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None

        n = arr.shape[0]
        if i < 0 or j < 0 or k < 0 or i >= n or j >= n or k >= n:
            return None

        try:
            p_i = arr[i]
            p_j = arr[j]
            p_k = arr[k]
            v1 = p_i - p_j
            v2 = p_k - p_j
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 <= 0.0 or n2 <= 0.0:
                return None
            cos_theta = float(np.dot(v1, v2) / (n1 * n2))
            if cos_theta > 1.0:
                cos_theta = 1.0
            if cos_theta < -1.0:
                cos_theta = -1.0
            return float(np.degrees(np.arccos(cos_theta)))
        except Exception:
            return None

    def _compute_residue_dihedral(
        self,
        viewer,
        object_id: str,
        i: int,
        j: int,
        k: int,
        l: int,
    ) -> Optional[float]:
        try:
            entry = viewer._objects.get(object_id)  # type: ignore[attr-defined]
        except Exception:
            return None
        if entry is None:
            return None

        state = getattr(entry, "state", None)
        coords = getattr(state, "coords", None)
        if coords is None:
            return None

        try:
            arr = np.asarray(coords, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None

        n = arr.shape[0]
        if (
            i < 0
            or j < 0
            or k < 0
            or l < 0
            or i >= n
            or j >= n
            or k >= n
            or l >= n
        ):
            return None

        try:
            p0 = arr[i]
            p1 = arr[j]
            p2 = arr[k]
            p3 = arr[l]

            b0 = p1 - p0
            b1 = p2 - p1
            b2 = p3 - p2

            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)
            if np.linalg.norm(n1) <= 0.0 or np.linalg.norm(n2) <= 0.0:
                return None

            n1_u = n1 / np.linalg.norm(n1)
            n2_u = n2 / np.linalg.norm(n2)
            b1_u = b1 / np.linalg.norm(b1) if np.linalg.norm(b1) > 0.0 else b1

            m1 = np.cross(n1_u, b1_u)
            x = float(np.dot(n1_u, n2_u))
            y = float(np.dot(m1, n2_u))
            return float(np.degrees(np.arctan2(y, x)))
        except Exception:
            return None

    def _cmd_objects(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        try:
            objects = viewer.list_objects()
        except Exception as exc:
            self._emit_error(f"Failed to list objects: {exc}")
            return

        if not objects:
            self._emit_message("No objects loaded.")
            return

        lines: List[str] = []
        for idx, obj in enumerate(objects, start=1):
            oid = obj.get("id", "?")
            name = obj.get("name", oid)
            visible = obj.get("visible", True)
            path = obj.get("source_path") or obj.get("path") or "?"
            vis_flag = "on" if visible else "off"
            lines.append(f"{idx}: {name} (id={oid}, visible={vis_flag}, path={path})")

        self._emit_message("Objects:\n" + "\n".join(lines))

    def _cmd_get_names(self, args: List[str]) -> None:
        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        try:
            objects = viewer.list_objects()
        except Exception as exc:
            self._emit_error(f"Failed to list objects: {exc}")
            return

        names = []
        for obj in objects:
            oid = str(obj.get("id", ""))
            name = str(obj.get("name") or oid)
            names.append(name)

        if not names:
            self._emit_message("[]")
        else:
            self._emit_message("[" + ", ".join(names) + "]")

    def _cmd_delete(self, args: List[str]) -> None:
        """Delete objects by id or name (PyMOL-style delete)."""

        if not args:
            self._emit_error("Usage: delete <object_name|id> [more ...]")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        removed: list[str] = []
        failed: list[str] = []

        for token in args:
            target = (token or "").strip()
            if not target:
                continue

            obj = self._find_object_by_name(viewer, target)
            if obj is None:
                failed.append(target)
                continue

            oid = str(obj.get("id", "")).strip()
            if not oid:
                failed.append(target)
                continue

            try:
                ok = bool(viewer.remove_object(oid))
            except Exception:
                ok = False

            if ok:
                removed.append(oid)
            else:
                failed.append(target)

        # Refresh UI/store if we have a window
        try:
            if window is not None:
                window._refresh_objects_from_viewer()
        except Exception:
            pass

        if removed:
            self._emit_message("Deleted: " + ", ".join(removed))
        if failed:
            self._emit_error("Not found or failed: " + ", ".join(failed))

    def _find_object_by_name(self, viewer, name: str) -> Optional[dict]:
        target = (name or "").strip().lower()
        if not target:
            return None
        try:
            objects = viewer.list_objects()
        except Exception:
            return None
        for obj in objects:
            obj_id = str(obj.get("id", ""))
            obj_name = str(obj.get("name", ""))
            if not obj_id and not obj_name:
                continue
            if target == obj_id.lower() or target == obj_name.lower():
                return obj
        return None

    def _cmd_enable(self, args: List[str]) -> None:
        self._cmd_enable_disable(args, visible=True)

    def _cmd_disable(self, args: List[str]) -> None:
        self._cmd_enable_disable(args, visible=False)

    def _cmd_enable_disable(self, args: List[str], *, visible: bool) -> None:
        if not args:
            self._emit_error("Usage: enable/disable <all|object_name>")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        target = (args[0] or "").strip()
        vis = bool(visible)

        if target.lower() in ("all", "*"):
            try:
                objects = viewer.list_objects()
            except Exception as exc:
                self._emit_error(f"Failed to list objects: {exc}")
                return
            for obj in objects:
                oid = obj.get("id")
                if not oid:
                    continue
                try:
                    if hasattr(window, "_set_object_visible"):
                        window._set_object_visible(str(oid), vis)
                    else:
                        viewer.set_object_visible(str(oid), vis)
                except Exception:
                    continue
            return

        obj_info = self._find_object_by_name(viewer, target)
        if obj_info is None:
            self._emit_error(f"Unknown object: {target}")
            return

        obj_id = str(obj_info.get("id"))
        try:
            if hasattr(window, "_set_object_visible"):
                window._set_object_visible(obj_id, vis)
            else:
                viewer.set_object_visible(obj_id, vis)
        except Exception as exc:
            action = "enable" if vis else "disable"
            self._emit_error(f"Failed to {action} object {target}: {exc}")

    def _cmd_select(self, args: List[str]) -> None:
        if not args:
            self._emit_error(
                "Usage: select [sel_name,] selection_expr | select sel_name"
            )
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        tokens = list(args)

        # Optional named-selection prefix: "sel1," expr
        sel_name: Optional[str] = None
        first = tokens[0]
        if first.endswith(","):
            sel_name = first[:-1].strip()
            tokens = tokens[1:]

        # Recall a previously defined named selection: "select sel1"
        if sel_name is None and len(tokens) == 1:
            key = tokens[0].strip().lower()
            if key in self._named_selections:
                self._apply_named_selection(key)
                return

        if not tokens:
            self._emit_error(
                "Usage: select [sel_name,] selection_expr | select sel_name"
            )
            return

        expr_text = " ".join(tokens).strip()
        if not expr_text:
            self._emit_error(
                "Usage: select [sel_name,] selection_expr | select sel_name"
            )
            return

        try:
            obj_id, obj_name, res_indices = self._resolve_selection_to_residue_indices(
                viewer, expr_text
            )
        except ValueError as exc:
            self._emit_error(str(exc))
            return

        if res_indices:
            try:
                viewer.set_selected_residues(res_indices, object_id=obj_id)
            except Exception as exc:
                self._emit_error(
                    f"Failed to select residues on {obj_name}: {exc}"
                )
                return
            if sel_name:
                key = sel_name.strip().lower()
                self._named_selections[key] = {
                    "object_id": obj_id,
                    "indices": list(res_indices),
                }
            self._emit_message(
                f"Selected {obj_name} residues "
                + ",".join(str(i + 1) for i in res_indices)
            )
        else:
            try:
                viewer.set_selected_residues([], object_id=obj_id)
            except Exception:
                pass
            if sel_name:
                key = sel_name.strip().lower()
                self._named_selections[key] = {
                    "object_id": obj_id,
                    "indices": [],
                }
            self._emit_message(f"Selected object {obj_name}")

    def _cmd_set(self, args: List[str]) -> None:
        if not args or len(args) < 2:
            self._emit_error("Usage: set <name> <value>")
            return

        name = (args[0] or "").strip().lower()
        value = " ".join(args[1:]).strip()
        if not name or not value:
            self._emit_error("Usage: set <name> <value>")
            return

        value_l = value.lower()
        bool_map = {
            "on": True,
            "off": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }

        if name in ("bg_color", "bg_colour"):
            self._cmd_bg_color([value])
            return

        if name in ("color_mode", "color"):
            self._cmd_color([value])
            return

        if name in (
            "cartoon",
            "trace",
            "ca_trace",
            "atoms",
            "sticks",
            "surface",
            "dots",
            "plane",
            "grid",
        ):
            if value_l not in bool_map:
                self._emit_error(
                    f"Value for set {name} must be one of: on, off, true, false, 1, 0"
                )
                return
            vis = bool_map[value_l]
            if vis:
                self._cmd_show([name])
            else:
                self._cmd_hide([name])
            return

        self._emit_error(
            "set command supports only bg_color, color_mode, and simple rep toggles "
            "(cartoon/trace/atoms/sticks/surface/dots/plane) at the moment."
        )

    def _apply_named_selection(self, name: str) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        entry = self._named_selections.get(name.lower())
        if not isinstance(entry, dict):
            self._emit_error(f"Unknown selection: {name}")
            return

        obj_id = str(entry.get("object_id", ""))
        if not obj_id:
            self._emit_error(f"Selection '{name}' has no associated object")
            return

        indices = entry.get("indices")
        if not isinstance(indices, list):
            indices = []

        # Ensure object still exists and is visible/active
        obj_info = self._find_object_by_name(viewer, obj_id)
        obj_name = obj_id
        if obj_info is not None:
            obj_name = str(obj_info.get("name") or obj_id)

        try:
            if hasattr(window, "_set_object_visible"):
                window._set_object_visible(obj_id, True)
            else:
                viewer.set_object_visible(obj_id, True)
        except Exception:
            pass

        try:
            if hasattr(window, "_select_object_in_ui"):
                window._select_object_in_ui(obj_id)
            else:
                viewer.set_active_object(obj_id)
        except Exception:
            pass

        try:
            viewer.set_selected_residues(indices, object_id=obj_id)
        except Exception:
            return

        if indices:
            self._emit_message(
                f"Recalled selection {name} on {obj_name} residues "
                + ",".join(str(i + 1) for i in indices)
            )
        else:
            self._emit_message(f"Recalled selection {name} on object {obj_name}")

    def _parse_residue_indices(
        self,
        expr: str,
        *,
        residue_numbers: Optional[np.ndarray] = None,
    ) -> Optional[List[int]]:
        """Parse a simple residue expression into 0-based indices.

        When ``residue_numbers`` is provided (PDB-style residue ids aligned with
        the CA trace), expressions like ``res 10-20`` are interpreted in terms of
        those residue numbers and may include negative values. Otherwise, the
        expression falls back to 1-based sequence indices (old behavior).

        Supported forms:

        - ``"5"``          -> single residue
        - ``"5-10"``       -> inclusive range
        - ``"-5"``         -> single negative residue number (with mapping)
        - ``"-5-10"``      -> range including negatives (with mapping)
        """

        text = (expr or "").strip()
        if not text:
            return None

        # Helper to map a single residue number to indices
        def _map_single(num: int) -> List[int]:
            if residue_numbers is not None:
                try:
                    arr = np.asarray(residue_numbers)
                except Exception:
                    return []
                if arr.ndim != 1 or arr.size == 0:
                    return []
                idx_list: List[int] = []
                for i, v in enumerate(arr):
                    try:
                        rv = int(v)
                    except Exception:
                        continue
                    if rv == num:
                        idx_list.append(int(i))
                return idx_list
            # Fallback: 1-based sequence index semantics
            if num <= 0:
                return []
            return [num - 1]

        # Single integer (positive or negative)
        m_single = re.fullmatch(r"(-?\d+)", text)
        if m_single:
            try:
                val = int(m_single.group(1))
            except Exception:
                return None
            indices = _map_single(val)
            return indices or None

        # Range "start-end" with optional negatives and optional spaces
        m_range = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", text)
        if m_range:
            try:
                start = int(m_range.group(1))
                end = int(m_range.group(2))
            except Exception:
                return None
            if start > end:
                start, end = end, start
            if residue_numbers is not None:
                try:
                    arr = np.asarray(residue_numbers)
                except Exception:
                    return None
                if arr.ndim != 1 or arr.size == 0:
                    return None
                idx_list: List[int] = []
                for i, v in enumerate(arr):
                    try:
                        rv = int(v)
                    except Exception:
                        continue
                    if start <= rv <= end:
                        idx_list.append(int(i))
                return idx_list or None
            # Fallback to 1-based sequence indices
            if end <= 0:
                return None
            out: List[int] = []
            for i in range(start, end + 1):
                if i > 0:
                    out.append(i - 1)
            return out or None

        return None

    def _parse_measurement_selections(
        self,
        args: List[str],
        *,
        expected_count: int,
        cmd: str,
    ) -> tuple[Optional[str], List[str]]:
        joined = " ".join(args).strip()
        pattern = ", ".join(f"sele{i + 1}" for i in range(expected_count))
        if not joined:
            raise ValueError(f"Usage: {cmd} {pattern}")

        parts = [part.strip() for part in joined.split(",") if part.strip()]
        if len(parts) == expected_count:
            return None, parts
        if len(parts) >= expected_count + 1:
            name = parts[0]
            return name, parts[1 : 1 + expected_count]

        raise ValueError(f"Usage: {cmd} {pattern}")

    def _resolve_selection_to_residue_indices(
        self,
        viewer,
        expr: str,
    ) -> tuple[str, str, List[int]]:
        text = (expr or "").strip()
        if not text:
            raise ValueError("Empty selection")

        try:
            tokens = shlex_split(text)
        except Exception as exc:
            raise ValueError(f"Could not parse selection {expr!r}: {exc}")

        if not tokens:
            raise ValueError("Empty selection")

        obj_info = None
        idx = 0
        first = tokens[0]
        t0 = first.lower()
        if t0 not in ("res", "resi", "residue", "name", "and", "or", "not"):
            obj_info = self._find_object_by_name(viewer, first)
            if obj_info is not None:
                idx = 1

        if obj_info is None:
            try:
                active_id = viewer.get_active_object_id()
            except Exception:
                active_id = None
            if active_id is None:
                raise ValueError("No active object for selection")
            obj_info = self._find_object_by_name(viewer, str(active_id))
            if obj_info is None:
                obj_info = {"id": active_id, "name": str(active_id)}

        obj_id = str(obj_info.get("id"))
        obj_name = str(obj_info.get("name") or obj_id)

        # Optional PDB residue numbers aligned with the CA trace. When
        # available, expressions like "res 10-20" are interpreted in terms of
        # these numbers instead of simple sequence indices.
        try:
            residue_numbers = viewer.get_residue_numbers(obj_id)
        except Exception:
            residue_numbers = None

        res_indices: List[int] = []
        atom_name: Optional[str] = None

        n_tokens = len(tokens)
        while idx < n_tokens:
            tok = tokens[idx].lower()
            if tok in ("and", "&"):
                idx += 1
                continue
            if tok in ("res", "resi", "residue"):
                if idx + 1 >= n_tokens:
                    raise ValueError("res expects an index or range")
                res_expr = tokens[idx + 1]
                partial = self._parse_residue_indices(
                    res_expr,
                    residue_numbers=residue_numbers,
                )
                if not partial:
                    raise ValueError(f"Invalid residue expression: {res_expr!r}")
                res_indices.extend(int(i) for i in partial)
                idx += 2
                continue
            if tok == "name":
                if idx + 1 >= n_tokens:
                    raise ValueError("name expects an atom name")
                atom_name = tokens[idx + 1]
                idx += 2
                continue
            raise ValueError(f"Unsupported selection token: {tokens[idx]!r}")

        # If no residue filter was given, this is an object-only selection.
        if not res_indices:
            return obj_id, obj_name, []

        # Normalize and deduplicate indices
        res_indices = sorted({int(i) for i in res_indices if int(i) >= 0})

        # Optionally filter by atom name, keeping only residues that contain that atom.
        if atom_name is not None:
            try:
                entry = viewer._objects.get(obj_id)  # type: ignore[attr-defined]
            except Exception:
                entry = None
            if entry is None:
                raise ValueError(f"Unknown object in selection: {obj_name}")

            state = getattr(entry, "state", None)
            atoms = getattr(state, "atoms", None)
            all_atom_res_ids = getattr(state, "all_atom_res_ids", None)
            residue_ids = getattr(state, "residue_ids", None)

            if atoms is None or all_atom_res_ids is None or residue_ids is None:
                raise ValueError(
                    f"Object {obj_name} does not expose atom-level coordinates for 'name' selections"
                )

            try:
                atom_res_ids_arr = np.asarray(all_atom_res_ids)
                res_ids_arr = np.asarray(residue_ids)
                atom_names_arr = np.char.strip(atoms["atom_name"].astype(str))
            except Exception:
                raise ValueError(f"Could not access atom data for object {obj_name}")

            name_norm = atom_name.strip()
            if name_norm:
                keep: List[int] = []
                for ri in res_indices:
                    if ri < 0 or ri >= res_ids_arr.shape[0]:
                        continue
                    rid = res_ids_arr[ri]
                    try:
                        mask = (atom_res_ids_arr == rid) & (
                            np.char.lower(atom_names_arr) == name_norm.lower()
                        )
                    except Exception:
                        continue
                    if np.any(mask):
                        keep.append(ri)

                if not keep:
                    raise ValueError(
                        f"No residues with atom named {name_norm!r} matched selection on {obj_name}"
                    )
                res_indices = sorted({int(i) for i in keep})

        # Validate indices against current coords if available.
        try:
            entry = viewer._objects.get(obj_id)  # type: ignore[attr-defined]
        except Exception:
            entry = None
        if entry is not None:
            state = getattr(entry, "state", None)
            coords = getattr(state, "coords", None)
            if coords is not None:
                try:
                    arr = np.asarray(coords, dtype=float)
                    n = int(arr.shape[0]) if arr.ndim == 2 else 0
                except Exception:
                    n = 0
                if n > 0:
                    res_indices = sorted(
                        {i for i in res_indices if 0 <= int(i) < n}
                    )

        return obj_id, obj_name, res_indices


    def _resolve_selection_to_atom(
        self,
        viewer,
        expr: str,
    ) -> tuple[str, str, int, Optional[str], np.ndarray]:
        text = (expr or "").strip()
        if not text:
            raise ValueError("Empty selection")

        try:
            tokens = shlex_split(text)
        except Exception as exc:
            raise ValueError(f"Could not parse selection {expr!r}: {exc}")

        if not tokens:
            raise ValueError("Empty selection")

        obj_info = None
        idx = 0
        first = tokens[0]
        t0 = first.lower()
        if t0 not in ("res", "resi", "residue", "name", "and", "or", "not"):
            obj_info = self._find_object_by_name(viewer, first)
            if obj_info is not None:
                idx = 1

        if obj_info is None:
            try:
                active_id = viewer.get_active_object_id()
            except Exception:
                active_id = None
            if active_id is None:
                raise ValueError("No active object for selection")
            obj_info = self._find_object_by_name(viewer, str(active_id))
            if obj_info is None:
                obj_info = {"id": active_id, "name": str(active_id)}

        obj_id = str(obj_info.get("id"))
        obj_name = str(obj_info.get("name") or obj_id)

        try:
            residue_numbers = viewer.get_residue_numbers(obj_id)
        except Exception:
            residue_numbers = None

        res_indices: Optional[List[int]] = None
        atom_name: Optional[str] = None

        n_tokens = len(tokens)
        while idx < n_tokens:
            tok = tokens[idx].lower()
            if tok in ("and", "&"):
                idx += 1
                continue
            if tok in ("res", "resi", "residue"):
                if idx + 1 >= n_tokens:
                    raise ValueError("res expects an index or range")
                res_expr = tokens[idx + 1]
                res_indices = self._parse_residue_indices(
                    res_expr,
                    residue_numbers=residue_numbers,
                )
                if not res_indices:
                    raise ValueError(f"Invalid residue expression: {res_expr!r}")
                idx += 2
                continue
            if tok == "name":
                if idx + 1 >= n_tokens:
                    raise ValueError("name expects an atom name")
                atom_name = tokens[idx + 1]
                idx += 2
                continue
            raise ValueError(f"Unsupported selection token: {tokens[idx]!r}")

        if res_indices is None:
            raise ValueError("Selection must specify residues using 'res'")
        if len(res_indices) != 1:
            raise ValueError("Selection must resolve to a single residue index")
        res_index = res_indices[0]
        try:
            entry = viewer._objects.get(obj_id)  # type: ignore[attr-defined]
        except Exception:
            entry = None
        if entry is None:
            raise ValueError(f"Unknown object in selection: {obj_name}")

        state = getattr(entry, "state", None)
        coords = getattr(state, "coords", None)
        if coords is None:
            raise ValueError(f"Object {obj_name} has no coordinates")

        try:
            ca_arr = np.asarray(coords, dtype=float)
        except Exception:
            raise ValueError(f"Could not access coordinates for object {obj_name}")

        if ca_arr.ndim != 2 or ca_arr.shape[1] != 3:
            raise ValueError(f"Invalid coordinate array for object {obj_name}")

        if res_index < 0 or res_index >= ca_arr.shape[0]:
            raise ValueError(
                f"Residue index {res_index + 1} out of range for object {obj_name}"
            )

        atoms = getattr(state, "atoms", None)
        all_atom_coords = getattr(state, "all_atom_coords", None)
        all_atom_res_ids = getattr(state, "all_atom_res_ids", None)
        residue_ids = getattr(state, "residue_ids", None)

        if atom_name is None:
            coord = ca_arr[res_index]
            return obj_id, obj_name, res_index, None, coord

        if (
            atoms is None
            or all_atom_coords is None
            or all_atom_res_ids is None
            or residue_ids is None
        ):
            raise ValueError(
                f"Object {obj_name} does not expose atom-level coordinates for 'name' selections"
            )

        try:
            atom_res_ids_arr = np.asarray(all_atom_res_ids)
            atom_coords_arr = np.asarray(all_atom_coords, dtype=float)
        except Exception:
            raise ValueError(f"Could not access atom coordinates for object {obj_name}")

        try:
            res_ids_arr = np.asarray(residue_ids)
        except Exception:
            raise ValueError(f"Could not access residue ids for object {obj_name}")

        if res_index < 0 or res_index >= res_ids_arr.shape[0]:
            raise ValueError(
                f"Residue index {res_index + 1} out of range for object {obj_name}"
            )

        target_res_id = res_ids_arr[res_index]

        try:
            atom_names_arr = np.char.strip(atoms["atom_name"].astype(str))
        except Exception:
            raise ValueError(f"Could not access atom names for object {obj_name}")

        name_norm = atom_name.strip()
        if not name_norm:
            coord = ca_arr[res_index]
            return obj_id, obj_name, res_index, None, coord

        try:
            mask = (atom_res_ids_arr == target_res_id) & (
                np.char.lower(atom_names_arr) == name_norm.lower()
            )
        except Exception:
            raise ValueError(f"Failed to match atom name {atom_name!r} on object {obj_name}")

        if not np.any(mask):
            raise ValueError(
                f"No atom named {name_norm!r} found at residue index {res_index + 1} on {obj_name}"
            )

        idx_arr = np.nonzero(mask)[0]
        atom_idx = int(idx_arr[0])
        if atom_idx < 0 or atom_idx >= atom_coords_arr.shape[0]:
            raise ValueError(
                f"Atom index out of range for object {obj_name}"
            )

        coord = atom_coords_arr[atom_idx]
        return obj_id, obj_name, res_index, name_norm, coord

    def _cmd_deselect(self, args: List[str]) -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        # Clear 3D residue selection on the active object
        try:
            viewer.set_selected_residues([])
        except Exception:
            pass

        # Also clear sequence selection in the UI if available
        try:
            seq_list = getattr(window, "seq_list", None)
            if seq_list is not None:
                seq_list.clearSelection()
        except Exception:
            pass

        self._emit_message("Deselected residues on active object")

    def _cmd_quit(self, args: List[str]) -> None:
        window = self.window
        if window is None:
            return
        try:
            window.close()
        except Exception:
            pass

    def _emit_message(self, text: str) -> None:
        callback = self._message_callback
        if callback is not None:
            callback(text)

    def _emit_error(self, text: str) -> None:
        callback = self._error_callback or self._message_callback
        if callback is not None:
            callback(text)
