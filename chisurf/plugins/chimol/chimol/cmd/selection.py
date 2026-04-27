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
        
        if name.startswith("metaball.") or name.startswith("metaball_"):
            prop = name.replace("metaball_", "").replace("metaball.", "")
            window, viewer = self._require_window_and_viewer()
            if viewer is None:
                return
            
            from ..config import _DISPLAY_CONFIG
            mcfg = _DISPLAY_CONFIG.setdefault("metaball", {})
            try:
                mcfg[prop] = float(value)
                viewer._update_view()
                self._emit_message(f"Metaball {prop} set to {value}")
            except ValueError:
                self._emit_error(f"Invalid value for metaball.{prop}: {value}")
            return

        self._emit_error(
            "set command supports bg_color, color_mode, rep toggles, "
            "and metaball properties (metaball.alpha, metaball.shininess, etc.)"
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

    def _resolve_selection_to_atom_mask(
        self,
        viewer,
        expr: str,
    ) -> tuple[str, str, np.ndarray]:
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
        first = tokens[0]
        t0 = first.lower()
        if t0 not in ("all", "none", "res", "resi", "residue", "name", "and", "or", "not", "within", "around", "expand", "byres", "bymol", "byobj"):
            obj_info = self._find_object_by_name(viewer, first)

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

        from .sele_parser import Evaluator, ParserError
        try:
            evaluator = Evaluator(viewer, obj_id)
            atom_mask = evaluator.evaluate(text)
        except ParserError as exc:
            raise ValueError(f"Selection parse error: {exc}")
        except NotImplementedError as exc:
            raise ValueError(f"Selection evaluation error: {exc}")

        return obj_id, obj_name, atom_mask


    def _resolve_selection_to_residue_indices(
        self,
        viewer,
        expr: str,
    ) -> tuple[str, str, List[int]]:
        text = (expr or "").strip()
        if not text:
            raise ValueError("Empty selection")

        # Basic object resolution (first token might be the object name if not a standard token)
        try:
            tokens = shlex_split(text)
        except Exception as exc:
            raise ValueError(f"Could not parse selection {expr!r}: {exc}")

        if not tokens:
            raise ValueError("Empty selection")

        obj_info = None
        first = tokens[0]
        t0 = first.lower()
        if t0 not in ("all", "none", "res", "resi", "residue", "name", "and", "or", "not", "within", "around", "expand", "byres", "bymol", "byobj"):
            obj_info = self._find_object_by_name(viewer, first)

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

        from .sele_parser import Evaluator, ParserError
        try:
            evaluator = Evaluator(viewer, obj_id)
            atom_mask = evaluator.evaluate(text)
        except ParserError as exc:
            raise ValueError(f"Selection parse error: {exc}")
        except NotImplementedError as exc:
            raise ValueError(f"Selection evaluation error: {exc}")

        if not np.any(atom_mask):
            return obj_id, obj_name, []

        try:
            entry = viewer._objects.get(obj_id)
            state = getattr(entry, "state", None)
            all_atom_res_ids = getattr(state, "all_atom_res_ids", None)
            residue_ids = getattr(state, "residue_ids", None)
            
            if all_atom_res_ids is None or residue_ids is None:
                raise ValueError(f"Object {obj_name} missing data for residue conversion")
                
            res_ids_arr = np.asarray(residue_ids)
            atom_res_ids_arr = np.asarray(all_atom_res_ids)
            
            # Find which globally unique residue IDs have at least one selected atom
            selected_res_ids = np.unique(atom_res_ids_arr[atom_mask])
            
            # Map selected global residue IDs back to their 0-based index in the 'residue_ids' array
            # We assume res_ids_arr contains ALL the unique global residue IDs in order.
            
            # Using np.isin and np.where
            mask = np.isin(res_ids_arr, selected_res_ids)
            res_indices = np.where(mask)[0].tolist()
            
        except Exception as exc:
            raise ValueError(f"Failed to extract residue indices: {exc}")

        return obj_id, obj_name, res_indices

    def _resolve_selection_to_atom(
        self,
        viewer,
        expr: str,
    ) -> tuple[str, str, int, Optional[str], np.ndarray]:
        text = (expr or "").strip()
        if not text:
            raise ValueError("Empty selection")

        # Basic object resolution
        try:
            tokens = shlex_split(text)
        except Exception as exc:
            raise ValueError(f"Could not parse selection {expr!r}: {exc}")

        if not tokens:
            raise ValueError("Empty selection")

        obj_info = None
        first = tokens[0]
        t0 = first.lower()
        if t0 not in ("all", "none", "res", "resi", "residue", "name", "and", "or", "not", "within", "around", "expand", "byres", "bymol", "byobj"):
            obj_info = self._find_object_by_name(viewer, first)

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

        from .sele_parser import Evaluator, ParserError
        try:
            evaluator = Evaluator(viewer, obj_id)
            atom_mask = evaluator.evaluate(text)
        except ParserError as exc:
            raise ValueError(f"Selection parse error: {exc}")
        except NotImplementedError as exc:
            raise ValueError(f"Selection evaluation error: {exc}")

        if not np.any(atom_mask):
            raise ValueError(f"Selection {expr!r} matched no atoms")

        # Pick the first matching atom
        try:
            entry = viewer._objects.get(obj_id)
            state = getattr(entry, "state", None)
            all_coords = getattr(state, "all_atom_coords", None)
            all_res_ids = getattr(state, "all_atom_res_ids", None)
            residue_ids = getattr(state, "residue_ids", None)
            atoms = getattr(state, "atoms", None)

            if all_coords is None:
                raise ValueError(f"Object {obj_name} missing coordinate data")

            # Get first index where mask is True
            first_idx = np.where(atom_mask)[0][0]
            
            coord = all_coords[first_idx]
            
            # Map back to residue index
            res_idx = -1
            if all_res_ids is not None and residue_ids is not None:
                rid = all_res_ids[first_idx]
                res_indices = np.where(residue_ids == rid)[0]
                if len(res_indices) > 0:
                    res_idx = int(res_indices[0])
            
            atom_name = None
            if atoms is not None:
                 atom_name = str(atoms["atom_name"][first_idx]).strip()

            return obj_id, obj_name, res_idx, atom_name, coord

        except Exception as exc:
            raise ValueError(f"Failed to resolve atom coordinate: {exc}")
