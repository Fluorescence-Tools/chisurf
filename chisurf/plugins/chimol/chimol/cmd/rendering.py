from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .base import BaseCmd
from ..colors import _PYMOL_COLORS, get_pymol_color


class RenderingMixin(BaseCmd):
    """Background, representation toggles, color modes and per-selection coloring."""

    # User-defined colors (via set_color command)
    _user_colors: Dict[str, np.ndarray] = {}

    def _mixin_commands(self):
        return {
            "bg_color": self._cmd_bg_color,
            "bg_colour": self._cmd_bg_color,
            "show": self._cmd_show,
            "hide": self._cmd_hide,
            "as": self._cmd_as,
            "center": self._cmd_center,
            "orient": self._cmd_orient,
            "zoom": self._cmd_zoom,
            "reset": self._cmd_reset,
            "get_view": self._cmd_get_view,
            "set_view": self._cmd_set_view,
            "color": self._cmd_color,
            "spectrum": self._cmd_spectrum,
            "cartoon": self._cmd_cartoon,
            "set_color": self._cmd_set_color,
            "get_color_index": self._cmd_get_color_index,
        }

    # ------------------------------------------------------------------ #
    # Background / rep toggles
    # ------------------------------------------------------------------ #
    def _cmd_bg_color(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: bg_color <color>")
            return
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return
        try:
            rgba = self._parse_color_spec(" ".join(args))
            viewer.set_background_color(rgba)
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
            self._emit_error("Usage: show/hide <cartoon|trace|atoms|sticks|dots|surface|metaball|plane>[, selection]")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        joined = " ".join(args).strip()
        rep_target = ""
        selection = None

        if "," in joined:
             parts = [p.strip() for p in joined.split(",", 1)]
             rep_target = parts[0].lower()
             selection = parts[1]
        else:
             rep_target = (args[0] or "").lower()
             if len(args) > 1:
                  selection = " ".join(args[1:])

        vis = bool(visible)

        if rep_target in ("all", "*"):
            # If selection given, maybe support it? For now, object-level visibility
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

        if selection:
            # 1. Handle cartoon/ribbon (residue-level)
            if rep_target in ("cartoon", "ribbon"):
                try:
                    obj_id, obj_name, res_indices = self._resolve_selection_to_residue_indices(
                        viewer, selection
                    )
                    entry = viewer._objects.get(obj_id)
                    mask = np.asarray(getattr(entry.state, "cartoon_mask"), dtype=bool).copy()
                    for ri in res_indices:
                         if 0 <= ri < mask.shape[0]:
                            mask[ri] = vis
                    entry.state.cartoon_mask = mask
                    viewer._update_view()
                    return
                except Exception as exc:
                    self._emit_error(str(exc))
                    return

            # 2. Handle balls/sticks (atom-level)
            if rep_target in ("atoms", "spheres", "balls", "ball", "sticks", "bonds"):
                try:
                    obj_id, obj_name, atom_mask = self._resolve_selection_to_atom_mask(
                        viewer, selection
                    )
                    entry = viewer._objects.get(obj_id)
                    field = "ball_mask" if rep_target not in ("sticks", "bonds") else "sticks_mask"
                    
                    cur_mask = getattr(entry.state, field)
                    if cur_mask is None or len(cur_mask) != len(atom_mask):
                         cur_mask = np.zeros(len(atom_mask), dtype=bool)
                    else:
                         cur_mask = cur_mask.copy()
                    
                    if vis:
                        cur_mask |= atom_mask
                    else:
                        cur_mask &= ~atom_mask
                    
                    setattr(entry.state, field, cur_mask)
                    viewer._update_view()
                    return
                except Exception as exc:
                    self._emit_error(str(exc))
                    return

        # Fallback to global representation toggle
        try:
            if rep_target in ("cartoon", "ribbon"):
                viewer.set_cartoon_visible(vis)
            elif rep_target in ("trace", "ca_trace", "lines"):
                viewer.set_trace_visible(vis)
            elif rep_target in ("atoms", "spheres", "balls", "ball"):
                viewer.set_atoms_visible_all(vis)
            elif rep_target in ("sticks", "bonds"):
                viewer.set_sticks_visible(vis)
            elif rep_target in ("dots", "points"):
                viewer.set_dots_visible(vis)
            elif rep_target in ("surface", "surf"):
                viewer.set_surface_visible(vis)
            elif rep_target in ("plane", "grid"):
                viewer.set_plane_visible(vis)
            elif rep_target in ("metaball", "metaballs", "mesh"):
                viewer.set_metaballs_visible(vis)
            else:
                self._emit_error(
                    f"Unsupported representation for show/hide: {rep_target}"
                )
                return
        except Exception as exc:
            self._emit_error(f"Failed to update representation '{rep_target}': {exc}")

    def _cmd_center(self, args: List[str]) -> None:
        """Center view on selection or all objects."""
        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        selection = " ".join(args).strip() or None
        if selection:
            try:
                # Note: this helper is available when mixed into Cmd
                obj_id, _, res_indices = self._resolve_selection_to_residue_indices(viewer, selection) # type: ignore
                if obj_id:
                    viewer.center(res_indices, object_id=obj_id)
                else:
                    self._emit_error(f"Selection '{selection}' did not resolve.")
            except Exception as exc:
                self._emit_error(f"Failed to center: {exc}")
        else:
            viewer.center()

    def _cmd_orient(self, args: List[str]) -> None:
        """Orient view on selection."""
        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        selection = " ".join(args).strip() or None
        if selection:
            try:
                obj_id, _, res_indices = self._resolve_selection_to_residue_indices(viewer, selection) # type: ignore
                if obj_id:
                    viewer.orient(res_indices, object_id=obj_id)
                else:
                    self._emit_error(f"Selection '{selection}' did not resolve.")
            except Exception as exc:
                self._emit_error(f"Failed to orient: {exc}")
        else:
            viewer.orient()

    def _cmd_zoom(self, args: List[str]) -> None:
        """Zoom view to fit selection."""
        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        # zoom [selection], [buffer]
        joined = " ".join(args)
        buffer = 2.0
        selection = None

        if "," in joined:
            parts = [p.strip() for p in joined.split(",", 1)]
            selection = parts[0] or None
            try:
                buffer = float(parts[1])
            except (ValueError, IndexError):
                pass
        else:
            selection = joined.strip() or None

        if selection:
            try:
                obj_id, _, res_indices = self._resolve_selection_to_residue_indices(viewer, selection) # type: ignore
                if obj_id:
                    viewer.zoom(res_indices, buffer=buffer, object_id=obj_id)
                else:
                    self._emit_error(f"Selection '{selection}' did not resolve.")
            except Exception as exc:
                self._emit_error(f"Failed to zoom: {exc}")
        else:
            viewer.zoom(buffer=buffer)

    def _cmd_reset(self, args: List[str]) -> None:
        """Reset view to default orientation and center."""
        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return
        viewer.reset_view()

    def _cmd_cartoon(self, args: List[str]) -> None:
        """Set cartoon display type similar to PyMOL ``cartoon``."""

        if not args:
            self._emit_error("Usage: cartoon <automatic|tube|oval|rect|arrow|loop> [, selection]")
            return
        mode = (args[0].rstrip(",") or "").strip().lower()
        from ..config import _DISPLAY_CONFIG
        cfg = _DISPLAY_CONFIG.setdefault("cartoon", {})
        if mode in ("automatic", "auto", "default", "oval", "rect", "rectangle", "arrow", "loop"):
            cfg["style"] = "ribbon"
        elif mode in ("tube", "trace"):
            cfg["style"] = "tube"
        else:
            self._emit_error(f"Unsupported cartoon type: {mode}")
            return
        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return
        try:
            viewer.set_cartoon_visible(True)
            viewer._update_view()
        except Exception:
            pass
        self._emit_message(f"Cartoon type set to {mode}")

    def _cmd_get_view(self, args: List[str]) -> str:
        """Return a copy/pasteable PyMOL-style set_view command."""

        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return ""
        try:
            vals = [float(v) for v in viewer.get_view_state()]
        except Exception as exc:
            self._emit_error(f"Failed to get view: {exc}")
            return ""
        return "set_view (" + ", ".join(f"{v:.9g}" for v in vals) + ")"

    def _cmd_set_view(self, args: List[str]) -> None:
        """Restore an 18-float view tuple."""

        joined = " ".join(args).strip()
        if not joined:
            self._emit_error("Usage: set_view (<18 floats>)")
            return
        text = joined.replace("set_view", " ").replace("(", " ").replace(")", " ")
        text = text.replace("\\", " ").replace(",", " ")
        try:
            vals = [float(tok) for tok in text.split()]
        except Exception:
            self._emit_error("set_view requires 18 numeric values")
            return
        if len(vals) != 18:
            self._emit_error("set_view requires 18 numeric values")
            return
        _, viewer = self._require_window_and_viewer()
        if viewer is None:
            return
        try:
            viewer.set_view_state(vals)
        except Exception as exc:
            self._emit_error(f"Failed to set view: {exc}")

    # ------------------------------------------------------------------ #
    # Color handling
    # ------------------------------------------------------------------ #
    def _cmd_color(self, args: List[str]) -> None:
        """Set a simple color mode or per-selection color."""

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

        # No comma: treat as simple color-mode toggle or uniform color on active object.
        raw = (args[0] or "").strip().lower()
        try:
            mode = self._normalize_color_mode(raw)
        except ValueError:
            # Not a mode name; try parsing as a color spec for uniform coloring.
            try:
                rgba = self._parse_color_spec(raw)
            except ValueError:
                self._emit_error(
                    f"Unrecognized color '{raw}'. Use a color name, #hex, "
                    "or one of: single, by_residue, by_ss, by_sequence, "
                    "by_element, by_chain, spectrum."
                )
                return
            # Apply uniform color to all residues via the comma form internally.
            self._cmd_color([raw + ", all"])
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

    def _cmd_spectrum(self, args: List[str]) -> None:
        """Color by a spectrum (rainbow). Alias for 'color spectrum'.

        PyMOL syntax: spectrum [expression [, palette [, selection]]]
        Chimol currently supports the global color mode part.
        """
        self._cmd_color(["spectrum"])

    def _cmd_set_color(self, args: List[str]) -> None:
        """Define a named color.

        PyMOL syntax: set_color name, [r, g, b] or set_color name, hex
        """
        if not args:
            self._emit_error("Usage: set_color <name>, <r,g,b> or <#hex>")
            return
        joined = " ".join(args).strip()
        if "," not in joined:
            self._emit_error("Usage: set_color <name>, <r,g,b> or <#hex>")
            return
        name_part, color_part = [p.strip() for p in joined.split(",", 1)]
        if not name_part or not color_part:
            self._emit_error("Usage: set_color <name>, <r,g,b> or <#hex>")
            return
        try:
            rgba = self._parse_color_spec(color_part)
        except (ValueError, KeyError) as exc:
            self._emit_error(f"Cannot parse color value: {exc}")
            return
        key = name_part.strip().lower()
        self._user_colors[key] = rgba
        self._emit_message(f"Defined color '{name_part.strip()}' = {rgba[:3]}")

    def _cmd_get_color_index(self, args: List[str]) -> Optional[int]:
        """Return the internal index for a named color (always 0 for compat)."""
        if not args:
            self._emit_error("Usage: get_color_index <name>")
            return None
        name = args[0].strip().lower()
        if name in _PYMOL_COLORS or name in self._user_colors:
            # PyMOL returns -1 for unknown colors; Chimol returns 0 for known.
            self._emit_message(f"Color index for '{name}': 0")
            return 0
        self._emit_error(f"Unknown color: {name}")
        return None

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
        if token in ("by_element", "element", "elem", "cpk", "by_elem"):
            return "by_element"
        if token in ("by_chain", "chain"):
            return "by_chain"
        if token in ("spectrum", "rainbow"):
            return "spectrum"
        raise ValueError(
            "Unsupported color mode. Use one of: "
            "single, by_residue, by_ss, by_sequence, by_element, by_chain, spectrum."
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

        # Check built-in PyMOL colors
        if name in _PYMOL_COLORS:
            r, g, b = _PYMOL_COLORS[name]
            return np.array([r, g, b, 1.0], dtype=float)

        # Check user-defined colors
        if name in self._user_colors:
            return self._user_colors[name].copy()

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
        obj_id, obj_name, atom_mask = self._resolve_selection_to_atom_mask(
            viewer, sele_expr
        )
        if not obj_id:
            raise ValueError("Selection did not resolve to an object")

        try:
            entry = viewer._objects.get(obj_id)
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
            n_atoms = int(np.asarray(all_atom_coords).shape[0])
            atom_mask = np.asarray(atom_mask, dtype=bool)
            if atom_mask.shape[0] != n_atoms:
                 # This shouldn't happen if Evaluator is correct
                 raise ValueError("Internal error: atom mask size mismatch")
        except Exception as exc:
            raise ValueError(f"Invalid atom data for coloring: {exc}")

        # Build/extend per-atom override array with NaN -> no override.
        try:
            cur_atom = np.asarray(state.colors_per_atom_override, dtype=float)
        except Exception:
            cur_atom = None
        if cur_atom is None or cur_atom.ndim != 2 or cur_atom.shape[0] != n_atoms:
            cur_atom = np.full((n_atoms, 4), np.nan, dtype=float)

        rgba4 = np.asarray(rgba, dtype=float).reshape(4)

        # Apply to atoms
        cur_atom[atom_mask, :] = rgba4
        state.colors_per_atom_override = cur_atom

        # Update per-residue override for residues where atoms were colored.
        # This keeps the cartoon view mostly consistent with the atom view.
        try:
            n_res = int(residue_ids.shape[0])
            cur_res = np.asarray(state.colors_per_residue_override, dtype=float)
        except Exception:
            cur_res = None
        if cur_res is None or cur_res.ndim != 2 or cur_res.shape[0] != n_res:
            cur_res = np.full((n_res, 4), np.nan, dtype=float)
            
        # Find which residue IDs have at least one colored atom
        res_ids_arr = np.asarray(residue_ids)
        atom_res_ids_arr = np.asarray(all_atom_res_ids)
        affected_rid = np.unique(atom_res_ids_arr[atom_mask])
        
        # Map affected global residue IDs to indices
        affected_res_mask = np.isin(res_ids_arr, affected_rid)
        cur_res[affected_res_mask, :] = rgba4
        state.colors_per_residue_override = cur_res

        try:
            viewer._update_view()
        except Exception:
            pass


