from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .command import Cmd


class RmfMixin:
    """Mixin for RMF/IHM specific commands."""

    def _cmd_rmf_hierarchy(self: Cmd, object_id: Optional[str] = None) -> None:
        """
        rmf_hierarchy [object_id]

        Print the RMF hierarchy for the given object to the console.
        """
        with self._viewer._activate_object(object_id):
            state = self._viewer._get_active_state()
            root = state.rmf_hierarchy
            if root is None:
                self._error("Object does not contain an RMF hierarchy.")
                return

            def print_node(node, indent=0):
                self._message("  " * indent + f"- {node.name} ({node.node_type})")
                for child in node.children:
                    print_node(child, indent + 1)

            self._message(f"RMF Hierarchy for {self._viewer.get_active_object_id()}:")
            print_node(root)

    def _cmd_rmf_readtraj(self: Cmd, path: str, object_id: Optional[str] = None) -> None:
        """
        rmf_readtraj path, [object_id]

        Read an RMF trajectory into an existing object.
        """
        # Placeholder for now, as load already handles RMF trajectories
        self._message("Use 'load' to read RMF files. rmf_readtraj is a placeholder.")

    def rmf_hierarchy(self, selection: Optional[str] = None) -> None:
        """Python-style RMF hierarchy command."""
        self._cmd_rmf_hierarchy(selection)
