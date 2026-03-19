from __future__ import annotations

from typing import List, Optional
from shlex import split as shlex_split
import numpy as np

from ..analysis.metrics import compute_rmsd, compute_kabsch
from .base import BaseCmd


class MeasurementMixin(BaseCmd):
    """Measurements, frames, and geometric helpers."""

    def _mixin_commands(self):
        return {
            "distance": self._cmd_distance,
            "angle": self._cmd_angle,
            "dihedral": self._cmd_dihedral,
            "rms": self._cmd_rms,
            "rms_cur": self._cmd_rms,  # alias
            "align": self._cmd_align,
            "super": self._cmd_super,
            "frame": self._cmd_frame,
            "frame_next": self._cmd_frame_next,
            "frame_prev": self._cmd_frame_prev,
        }

    def _add_measurement(self, viewer, name: str, kind: str, positions: np.ndarray, label: str):
        if not name:
            name = f"{kind}_{len(viewer._measurements)}"
        
        mdata = {
            "kind": kind,
            "positions": positions,
            "label": label,
            "color": [1.0, 1.0, 0.0, 1.0]
        }
        
        cur = dict(viewer._measurements)
        cur[name] = mdata
        viewer._measurements = cur
        viewer._update_view()

    # ------------------------------------------------------------------ #
    # Frames
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Measurements
    # ------------------------------------------------------------------ #
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

        dist_val = f"{dist:.3f}"
        self._emit_message(f"{prefix}{label1} - {label2}: {dist_val}")
        
        positions = np.array([v1, v2])
        self._add_measurement(viewer, meas_name, "distance", positions, dist_val)

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

        val_str = f"{value:.3f}"
        self._emit_message(
            f"{prefix}{label1} - {label2} - {label3}: {val_str}"
        )
        
        positions = np.array([v1, v2, v3])
        self._add_measurement(viewer, meas_name, "angle", positions, val_str)

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

        val_str = f"{value:.3f}"
        self._emit_message(
            prefix + f"{label1} - {label2} - {label3} - {label4}: {val_str}"
        )
        
        positions = np.array([v1, v2, v3, v4])
        self._add_measurement(viewer, meas_name, "dihedral", positions, val_str)

    # ------------------------------------------------------------------ #
    # RMS / Align
    # ------------------------------------------------------------------ #
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

    def _cmd_align(self, args: List[str]) -> None:
        """Usage: align mobile_selection, target_selection [, cutoff [, cycles]]"""
        self._cmd_align_or_super(args, cmd="align")

    def _cmd_super(self, args: List[str]) -> None:
        """Usage: super mobile_selection, target_selection [, cutoff [, cycles]]"""
        self._cmd_align_or_super(args, cmd="super")

    def _cmd_align_or_super(self, args: List[str], cmd: str = "align") -> None:
        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        if not args:
            self._emit_error(f"Usage: {cmd} mobile_selection, target_selection [, cutoff [, cycles]]")
            return

        # Parse arguments: mobile, target [, cutoff [, cycles]]
        joined = " ".join(args).strip()
        raw_parts = [p.strip() for p in joined.split(",") if p.strip()]
        
        if len(raw_parts) < 2:
            raw_parts = shlex_split(joined)
            if len(raw_parts) < 2:
                self._emit_error(f"Usage: {cmd} mobile_selection, target_selection")
                return

        mobile_expr = raw_parts[0]
        target_expr = raw_parts[1]
        
        # Default values
        cutoff = 2.0
        cycles = 5
        
        # Parse extra parts as either positional or keyword
        for i, part in enumerate(raw_parts[2:]):
             if "=" in part:
                  key, val = [p.strip() for p in part.split("=", 1)]
                  if key.lower() == "cutoff":
                       try: cutoff = float(val)
                       except ValueError: pass
                  elif key.lower() == "cycles":
                       try: cycles = int(val)
                       except ValueError: pass
             else:
                  # Positional fallback
                  if i == 0: # cutoff
                       try: cutoff = float(part)
                       except ValueError: pass
                  elif i == 1: # cycles
                       try: cycles = int(part)
                       except ValueError: pass

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

        # For super, we might want to match by name if indices differ?
        # For now, let's assume sequence-based matching (by index in the selection)
        # but only if number of residues is compatible.
        
        try:
            mob_coords = viewer.get_residue_positions(
                mob_indices if mob_indices else None, object_id=mob_obj
            )
            tgt_coords = viewer.get_residue_positions(
                tgt_indices if tgt_indices else None, object_id=tgt_obj
            )
        except Exception as exc:
            self._emit_error(f"Failed to access coordinates for {cmd}: {exc}")
            return

        if mob_coords.size == 0 or tgt_coords.size == 0:
            self._emit_error("Selections must contain at least one residue (CA)")
            return

        count = min(mob_coords.shape[0], tgt_coords.shape[0])
        if count < 3:
            self._emit_error(f"{cmd} requires at least three residues in each selection")
            return

        # Subset to matching count
        m_coords = mob_coords[:count]
        t_coords = tgt_coords[:count]
        
        # Iterative outlier rejection
        current_mask = np.ones(count, dtype=bool)
        final_rmsd = 0.0
        final_rot = np.eye(3)
        final_trans = np.zeros(3)
        final_count = count

        for i in range(cycles + 1):
            subset_count = np.sum(current_mask)
            if subset_count < 3:
                break
                
            m_sub = m_coords[current_mask]
            t_sub = t_coords[current_mask]
            
            try:
                rot, trans, rmsd = compute_kabsch(m_sub, t_sub)
            except ValueError as exc:
                self._emit_error(f"Fit failed on cycle {i}: {exc}")
                return
            
            final_rmsd = rmsd
            final_rot = rot
            final_trans = trans
            final_count = int(subset_count)

            if i < cycles:
                # Calculate all distances after this fit
                m_aligned = (m_coords @ rot) + trans
                dists = np.linalg.norm(m_aligned - t_coords, axis=1)
                new_mask = dists <= cutoff
                
                # If no change in mask, we converged
                if np.array_equal(new_mask, current_mask):
                    break
                
                # Ensure we have enough points left
                if np.sum(new_mask) < 3:
                    # Maybe too aggressive? Keep top 50%?
                    sorted_indices = np.argsort(dists)
                    new_mask = np.zeros(count, dtype=bool)
                    half = max(3, count // 2)
                    new_mask[sorted_indices[:half]] = True
                
                current_mask = new_mask
            else:
                break

        try:
            viewer.apply_transform_to_object(final_rot.T, final_trans, object_id=mob_obj)
        except Exception as exc:
            self._emit_error(f"Failed to apply {cmd} transform: {exc}")
            return

        self._emit_message(
            f"{cmd.capitalize()}: aligned {mob_name} onto {tgt_name} using {final_count}/{count} atoms "
            f"(RMSD: {final_rmsd:.3f} Å)"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
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
        """Return per-atom coordinates for the given residue indices."""

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
