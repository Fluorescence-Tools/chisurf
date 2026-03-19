from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING
import copy
from dataclasses import dataclass, field

from .base import BaseCmd

if TYPE_CHECKING:
    from ..renderer.view import MolView

class EditingMixin(BaseCmd):
    def _mixin_commands(self):
        return {
            "iterate": self._cmd_iterate,
            "alter": self._cmd_alter,
            "remove": self._cmd_remove,
            "delete": self._cmd_remove,  # alias
            "pseudoatom": self._cmd_pseudoatom,
        }

    def _cmd_pseudoatom(self, args: List[str]) -> None:
        """Usage: pseudoatom name [, selection [, label [, pos [, b [, q [, color [, state [, mode [, quiet ]]]]]]]]]"""
        if not args:
            self._emit_error("Usage: pseudoatom name, [selection, [label, [pos, ...]]]")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        # Improved parsing to handle named arguments and commas in lists
        joined = " ".join(args)
        
        # Split by comma but be careful with brackets (pos=[1,2,3])
        parts = []
        current = ""
        bracket_level = 0
        for char in joined:
             if char == '[': bracket_level += 1
             elif char == ']': bracket_level -= 1
             elif char == ',' and bracket_level == 0:
                  parts.append(current.strip())
                  current = ""
                  continue
             current += char
        parts.append(current.strip())
        
        # Dictionary of possible parameters
        params = {
            "name": None,
            "selection": "none",
            "label": "",
            "pos": None,
            "b": 0.0,
            "q": 1.0,
            "color": None,
            "state": 0,
            "mode": None,
            "quiet": True
        }
        
        positional_names = ["name", "selection", "label", "pos", "b", "q", "color", "state", "mode", "quiet"]
        
        for i, part in enumerate(parts):
             if "=" in part:
                  key, val = [p.strip() for p in part.split("=", 1)]
                  if key in params:
                       params[key] = val
             elif i < len(positional_names):
                  params[positional_names[i]] = part

        name = params["name"]
        if not name:
             self._emit_error("pseudoatom name is required")
             return
             
        selection = params["selection"]
        label = params["label"]
        pos_val = params["pos"]
        b_factor = float(params["b"]) if params["b"] is not None else 0.0
        occupancy = float(params["q"]) if params["q"] is not None else 1.0

        pos: Optional[List[float]] = None
        if isinstance(pos_val, str):
             if pos_val.startswith("[") and pos_val.endswith("]"):
                  try:
                       pos = [float(x) for x in pos_val[1:-1].replace(",", " ").split()]
                  except Exception:
                       pos = None
        elif isinstance(pos_val, (list, tuple)) and len(pos_val) == 3:
             pos = [float(x) for x in pos_val]
        
        # If selection center logic
        if not pos and selection.lower() != "none" and selection:
             try:
                  obj_id, _, mask = self._resolve_selection_to_atom_mask(viewer, selection)
                  entry = viewer._objects.get(obj_id)
                  if entry and entry.state.atoms is not None:
                       coords = entry.state.all_atom_coords[mask]
                       if coords.size > 0:
                            pos = np.mean(coords, axis=0).tolist()
             except Exception:
                  pass

        if not pos:
             pos = [0.0, 0.0, 0.0]

        # In ChiMol, we often want to add this to a new object or an existing one.
        # PyMOL adds it to 'name' object. If 'name' exists, it appends an atom.
        
        obj_info = self._find_object_by_name(viewer, name)
        if obj_info is None:
             # Create new object with one atom
             @dataclass
             class DummyStructure:
                  atoms: np.ndarray
                  xyz: np.ndarray

             atom_dtype = [
                 ('xyz', 'f4', (3,)),
                 ('atom_name', 'S10'),
                 ('res_id', 'i4'),
                 ('res_name', 'S10'),
                 ('chain_id', 'S4'),
                 ('element', 'S2'),
                 ('b_factor', 'f4'),
                 ('occupancy', 'f4')
             ]
             
             data = np.zeros(1, dtype=atom_dtype)
             data[0]['xyz'] = pos
             data[0]['atom_name'] = b'PS1'
             data[0]['res_id'] = 1
             data[0]['res_name'] = b'PSD'
             data[0]['chain_id'] = b' '
             data[0]['element'] = b'Ps'
             data[0]['b_factor'] = b_factor
             data[0]['occupancy'] = occupancy
             
             struct = DummyStructure(atoms=data, xyz=data['xyz'])
             
             # Need to create object via window if possible to get registry/etc.
             if window is not None and hasattr(window, "_load_structure_from_path"):
                  # This is a bit hacky, but MolView doesn't easily create objects from memory via commands yet.
                  # Let's use viewer directly and hope the window refreshes.
                  oid = viewer._create_object(name=name)
                  viewer.set_active_object(oid)
                  viewer.set_structure(struct)
                  window._refresh_objects_from_viewer()
             else:
                  oid = viewer._create_object(name=name)
                  viewer.set_active_object(oid)
                  viewer.set_structure(struct)
        else:
             # Append to existing object
             oid = str(obj_info['id'])
             entry = viewer._objects.get(oid)
             if entry and entry.state.atoms is not None:
                  old_atoms = entry.state.atoms
                  new_atom = np.zeros(1, dtype=old_atoms.dtype)
                  for f in old_atoms.dtype.names:
                       if f == 'xyz': new_atom[0][f] = pos
                       elif f == 'atom_name': new_atom[0][f] = b'PS1'
                       elif f == 'res_id': 
                            if len(old_atoms) > 0: 
                                 new_atom[0][f] = np.max(old_atoms['res_id']) + 1
                            else:
                                 new_atom[0][f] = 1
                       elif f == 'res_name': new_atom[0][f] = b'PSD'
                       elif f == 'b_factor': new_atom[0][f] = b_factor
                       elif f == 'occupancy': new_atom[0][f] = occupancy
                       else:
                            # default to what's in first atom or zero/empty
                            if len(old_atoms) > 0:
                                 new_atom[0][f] = old_atoms[0][f]
                  
                  entry.state.atoms = np.concatenate([old_atoms, new_atom])
                  entry.state.all_atom_coords = entry.state.atoms['xyz'].copy()
                  
                  # Re-run set_structure logic to update trace/masks
                  viewer.set_structure(entry.state)

        self._emit_message(f"Created pseudoatom {name} at {pos}")

    def _cmd_iterate(self, args: List[str]) -> None:
        """Usage: iterate selection, expression"""
        self._cmd_alter_or_iterate(args, read_only=True)

    def _cmd_alter(self, args: List[str]) -> None:
        """Usage: alter selection, expression"""
        self._cmd_alter_or_iterate(args, read_only=False)

    def _cmd_alter_or_iterate(self, args: List[str], read_only: bool) -> None:
        if len(args) < 2:
            cmd = "iterate" if read_only else "alter"
            self._emit_error(f"Usage: {cmd} selection, expression")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        joined = " ".join(args)
        if "," not in joined:
            self._emit_error("Selection and expression must be separated by a comma")
            return
        
        sele_expr, python_expr = [p.strip() for p in joined.split(",", 1)]

        try:
            obj_id, obj_name, atom_mask = self._resolve_selection_to_atom_mask(
                viewer, sele_expr
            )
        except Exception as exc:
            self._emit_error(str(exc))
            return

        entry = viewer._objects.get(obj_id)
        if entry is None or entry.state.atoms is None:
            self._emit_error(f"Object {obj_name} has no atoms")
            return

        atoms = entry.state.atoms
        indices = np.nonzero(atom_mask)[0]
        if indices.size == 0:
            return

        # Prepare namespace
        # We need to map field names to friendly names
        field_map = {
            "atom_name": "name",
            "res_name": "resn",
            "res_id": "resi",
            "chain_id": "chain",
            "element": "elem",
            "b_factor": "b",
            "occupancy": "q",
        }
        
        # Reverse map for alter
        reverse_map = {v: k for k, v in field_map.items()}

        count = 0
        try:
            # Compiled expression for speed if many atoms
            code = compile(python_expr, "<string>", "exec")
            
            for idx in indices:
                atom = atoms[idx]
                namespace = {}
                
                # Load current values
                for f, alias in field_map.items():
                    if f in atoms.dtype.names:
                        val = atom[f]
                        if isinstance(val, (bytes, np.bytes_)):
                             val = val.decode()
                        namespace[alias] = val
                
                xyz = atom["xyz"]
                namespace["x"] = float(xyz[0])
                namespace["y"] = float(xyz[1])
                namespace["z"] = float(xyz[2])
                
                exec(code, {}, namespace)
                
                if not read_only:
                    # Save changed values
                    for alias, f in reverse_map.items():
                        if alias in namespace and f in atoms.dtype.names:
                            val = namespace[alias]
                            # Handle types (int, float, string)
                            target_dtype = atoms.dtype[f]
                            if target_dtype.kind in ('S', 'U'):
                                if isinstance(val, str):
                                    atom[f] = val.encode() if target_dtype.kind == 'S' else val
                            else:
                                atom[f] = val
                    
                    # Coordinates
                    new_x = namespace.get("x", xyz[0])
                    new_y = namespace.get("y", xyz[1])
                    new_z = namespace.get("z", xyz[2])
                    atom["xyz"] = [new_x, new_y, new_z]
                
                count += 1
                
        except Exception as exc:
            self._emit_error(f"Error during execution: {exc}")
            return

        if not read_only:
            # If we altered, we need to notify the viewer to rebuild
            # Actually, the 'atoms' array in state might be the same object
            # but we should re-trigger updates.
            # We might also need to update all_atom_coords if that was cached separately
            if "xyz" in atoms.dtype.names:
                 entry.state.all_atom_coords = atoms["xyz"].copy()
            
            viewer._update_view()

        verb = "Iterated over" if read_only else "Altered"
        self._emit_message(f"{verb} {count} atoms")

    def _cmd_remove(self, args: List[str]) -> None:
        """Usage: remove selection"""
        if not args:
            self._emit_error("Usage: remove selection")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        selection = " ".join(args)
        try:
            obj_id, obj_name, atom_mask = self._resolve_selection_to_atom_mask(
                viewer, selection
            )
        except Exception as exc:
            self._emit_error(str(exc))
            return

        entry = viewer._objects.get(obj_id)
        if entry is None or entry.state.atoms is None:
            return

        keep_mask = ~atom_mask
        if np.all(keep_mask):
            return

        if not np.any(keep_mask):
            # Remove entire object? Or just clear it?
            # PyMOL usually keeps the object but it's empty.
            # Here we'll just clear atoms.
            entry.state.atoms = np.array([], dtype=entry.state.atoms.dtype)
            entry.state.all_atom_coords = None
        else:
            entry.state.atoms = entry.state.atoms[keep_mask].copy()
            entry.state.all_atom_coords = entry.state.atoms["xyz"].copy()

        # Update masks if they exist
        if entry.state.ball_mask is not None:
             if len(entry.state.ball_mask) == len(keep_mask):
                  entry.state.ball_mask = entry.state.ball_mask[keep_mask].copy()
        
        if entry.state.sticks_mask is not None:
             if len(entry.state.sticks_mask) == len(keep_mask):
                  entry.state.sticks_mask = entry.state.sticks_mask[keep_mask].copy()

        # Rebuild trace if needed? MolView.set_structure does a lot of work.
        # For now, just trigger view update.
        # NOTE: Full re-processing might be needed if CA atoms were removed.
        # We might want to call a method like viewer.update_from_atoms(obj_id)
        
        # A hack for now: tell MolView to re-process the atoms
        viewer.set_structure(entry.state) 
        
        self._emit_message(f"Removed {np.sum(atom_mask)} atoms from {obj_name}")
