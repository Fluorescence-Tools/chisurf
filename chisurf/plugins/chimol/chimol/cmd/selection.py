from __future__ import annotations

import re
from typing import List, Optional
from shlex import split as shlex_split

import numpy as np

from .base import BaseCmd


class SelectionMixin(BaseCmd):
    """Selection handling, object visibility, and simple set/enable toggles."""

    def _mixin_commands(self):
        return {
            "select": self._cmd_select,
            "set": self._cmd_set,
            "enable": self._cmd_enable,
            "disable": self._cmd_disable,
            "deselect": self._cmd_deselect,
            "objects": self._cmd_objects,
            "get_names": self._cmd_get_names,
        }

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Helpers used by other command groups
    # ------------------------------------------------------------------ #
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
        """Parse a simple residue expression into 0-based indices."""

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
        m_range = re.fullmatch(r"(-?\d+)\\s*-\\s*(-?\\d+)", text)
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

        # Optional PDB residue numbers aligned with the CA trace.
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
