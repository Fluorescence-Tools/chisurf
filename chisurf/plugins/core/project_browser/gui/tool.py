from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtWidgets

from chisurf import logging

from .client import ProjectBrowserClient


class SaveProjectDialog(QtWidgets.QDialog):
    """Single dialog for project name, visibility, and notes."""

    def __init__(
        self,
        current_name: str = "",
        parent: QtWidgets.QWidget | None = None,
        allow_name_edit: bool = True,
        title: str = "Save Project",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self._name_edit = QtWidgets.QLineEdit(current_name)
        self._name_edit.setPlaceholderText("Enter project name")
        self._name_edit.setEnabled(allow_name_edit)
        form.addRow("Project name:", self._name_edit)

        self._vis_combo = QtWidgets.QComboBox()
        self._vis_combo.addItems(["Private", "Public"])
        self._vis_combo.setToolTip(
            "Private: only the owner can see this project.\n"
            "Public: any authenticated user can see this project."
        )
        form.addRow("Visibility:", self._vis_combo)

        self._notes_edit = QtWidgets.QPlainTextEdit()
        self._notes_edit.setPlaceholderText("Optional notes...")
        self._notes_edit.setMaximumHeight(120)
        form.addRow("Notes:", self._notes_edit)

        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def project_name(self) -> str:
        """Return the project name entered in the dialog."""
        return self._name_edit.text().strip()

    @property
    def visibility(self) -> str:
        """Return the selected visibility as a lowercase string."""
        return self._vis_combo.currentText().lower()

    @property
    def notes(self) -> str:
        """Return optional notes entered in the dialog."""
        return self._notes_edit.toPlainText().strip()


class CollisionDialog(QtWidgets.QDialog):
    """Dialog to show ID collisions before import."""

    def __init__(
        self,
        collisions: dict[str, list[str]],
        origin: dict[str, Any],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Import — Collision Warning")
        self.setModal(True)
        self.setMinimumSize(500, 300)
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel(
            f"The following IDs from the archive already exist in the database.\n"
            f"Original project: {origin.get('project_id', '?')} "
            f"v{origin.get('version_number', '?')}.\n"
            "On confirmation, conflicting IDs will be remapped."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        lines = []
        for category, ids in collisions.items():
            if ids:
                lines.append(f"<b>{category}</b> ({len(ids)}):")
                for cid in ids[:20]:
                    lines.append(f"  - {cid}")
                if len(ids) > 20:
                    lines.append(f"  ... and {len(ids)-20} more")
                lines.append("")
        text.setHtml("<br>".join(lines) or "<i>No collisions found</i>")
        layout.addWidget(text)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Remap & Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ProjectBrowserTool(QtWidgets.QMainWindow):
    """Browser for MFDB-backed project versions."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Open Project")
        self.setMinimumSize(800, 500)

        self._client: ProjectBrowserClient | None = None
        self._projects: list[dict[str, Any]] = []

        self._init_ui()
        self._connect_signals()
        self.refresh()

    def _make_client(self) -> ProjectBrowserClient:
        return ProjectBrowserClient(inprocess=True)

    @property
    def client(self) -> ProjectBrowserClient:
        """Return the in-process project browser RPC client."""
        if self._client is None:
            self._client = self._make_client()
        return self._client

    @staticmethod
    def _project_counts_from_payload(payload: Any) -> tuple[int, int]:
        data = payload.to_dict() if hasattr(payload, "to_dict") else payload
        if not isinstance(data, dict):
            return 0, 0
        datasets = data.get("datasets", {})
        fits = data.get("fits", [])
        dataset_count = len(datasets) if hasattr(datasets, "__len__") else 0
        fit_count = len(fits) if hasattr(fits, "__len__") else 0
        return fit_count, dataset_count

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self._action_toolbar = self.addToolBar("Project Actions")
        self._action_toolbar.setObjectName("ProjectBrowserActions")
        self._action_toolbar.setMovable(False)

        self._open_btn = self._add_toolbar_button(
            "📂 Open / Restore",
            "Restore the selected project version or newest version of the selected project",
            self._on_open,
        )
        self._action_toolbar.addSeparator()
        self._save_btn = self._add_toolbar_button(
            "💾 Save Current Project",
            "Save the current Chisurf project as a new version in the database",
            self._on_save,
        )
        self._export_btn = self._add_toolbar_button(
            "📤 Export .csp",
            "Export the selected version to a .csp archive file",
            self._on_export,
        )
        self._import_btn = self._add_toolbar_button(
            "📥 Import Project",
            "Import a .csp archive into the database",
            self._on_import,
        )
        self._delete_btn = self._add_toolbar_button(
            "🗑️ Delete Version",
            "Delete the selected version (requires manage permission)",
            self._on_delete,
        )
        self._action_toolbar.addSeparator()
        self._refresh_btn = self._add_toolbar_button(
            "🔄 Refresh",
            "Refresh the project list",
            self.refresh,
        )

        # -- Search/filter controls --
        toolbar = QtWidgets.QHBoxLayout()

        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("Search projects...")
        self._search_edit.setClearButtonEnabled(True)
        toolbar.addWidget(self._search_edit, 1)

        self._show_public_cb = QtWidgets.QCheckBox("Show public")
        self._show_public_cb.setChecked(True)
        self._show_public_cb.setToolTip("Include public-visible projects in the list")
        toolbar.addWidget(self._show_public_cb)

        layout.addLayout(toolbar)

        # -- Tree --
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels([
            "Project / Version", "ID", "Owner", "Status", "Visibility",
            "Datasets", "Fits", "Created", "Notes",
        ])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setAnimated(True)
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._tree.setSortingEnabled(True)
        layout.addWidget(self._tree, 1)

    def _add_toolbar_button(
        self,
        text: str,
        tooltip: str,
        slot: Any,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self._action_toolbar)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        button.clicked.connect(slot)
        self._action_toolbar.addWidget(button)
        return button

    def _connect_signals(self) -> None:
        self._refresh_btn.clicked.connect(self.refresh)
        self._search_edit.textChanged.connect(self._on_search)
        self._show_public_cb.toggled.connect(self.refresh)
        self._open_btn.clicked.connect(self._on_open)
        self._save_btn.clicked.connect(self._on_save)
        self._export_btn.clicked.connect(self._on_export)
        self._import_btn.clicked.connect(self._on_import)
        self._delete_btn.clicked.connect(self._on_delete)
        self._tree.itemDoubleClicked.connect(self._on_open)

    def refresh(self) -> None:
        """Reload projects from the database and rebuild the tree view."""
        try:
            self._projects = self.client.list_projects(
                show_public=self._show_public_cb.isChecked(),
                search=self._search_edit.text().strip() or None,
            )
            self._populate_tree()
        except Exception as exc:
            logging.error("Failed to list projects: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to list projects:\n{exc}")

    def _populate_tree(self) -> None:
        self._tree.clear()
        for proj in self._projects:
            pid = proj.get("project_id", "?")
            pname = proj.get("project_name", "(unnamed)")
            owner = proj.get("owner_user_id", "")
            vis = proj.get("visibility", "private")
            vcount = proj.get("version_count", 0)
            created = proj.get("created_at", "")[:19].replace("T", " ")

            parent = QtWidgets.QTreeWidgetItem(self._tree)
            parent.setText(0, f"{pname}  ({vcount} versions)")
            parent.setText(1, pid)
            parent.setText(2, owner)
            parent.setText(4, vis)
            parent.setText(7, created)
            parent.setData(0, QtCore.Qt.UserRole, proj)
            parent.setToolTip(0, f"Project: {pid}\nOwner: {owner}\nVersions: {vcount}")

            for ver in proj.get("versions", []):
                child = QtWidgets.QTreeWidgetItem(parent)
                vn = ver.get("version_number", "?")
                child.setText(0, f"v{vn}  {ver.get('project_name', pname)}")
                child.setText(1, ver.get("version_id", ""))
                child.setText(2, ver.get("owner_user_id", ""))
                child.setText(3, ver.get("status", ""))
                child.setText(5, str(ver.get("dataset_count", 0)))
                child.setText(6, str(ver.get("fit_count", 0)))
                child.setText(7, (ver.get("created_at", "") or "")[:19].replace("T", " "))
                child.setText(8, (ver.get("notes", "") or "")[:60])
                child.setData(0, QtCore.Qt.UserRole, ver)
                child.setToolTip(0, (
                    f"Version: {ver.get('version_id', '')}\n"
                    f"Project: {ver.get('project_id', '')}\n"
                    f"Owner: {ver.get('owner_user_id', '')}\n"
                    f"Number: v{vn}\n"
                    f"Created: {ver.get('created_at', '')}\n"
                    f"Datasets: {ver.get('dataset_count', 0)}  Fits: {ver.get('fit_count', 0)}"
                ))

            parent.setExpanded(False)

        for i in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(i)

    def _selected_version(self) -> dict[str, Any] | None:
        items = self._tree.selectedItems()
        for item in items:
            data = item.data(0, QtCore.Qt.UserRole)
            if data and data.get("version_id"):
                return data
        return None

    def _selected_restore_version(self) -> dict[str, Any] | None:
        item = self._tree.currentItem()
        if item is not None:
            data = item.data(0, QtCore.Qt.UserRole)
            if isinstance(data, dict):
                if data.get("version_id"):
                    return data
                if data.get("latest_version_id"):
                    for ver in data.get("versions", []):
                        if ver.get("version_id") == data.get("latest_version_id"):
                            return ver
                    if data.get("versions"):
                        return data["versions"][0]
        return self._selected_version()

    def _on_search(self, text: str) -> None:
        self.refresh()

    def _on_open(self) -> None:
        ver = self._selected_restore_version()
        if not ver:
            QtWidgets.QMessageBox.information(self, "Select Project", "Please select a project or project version to restore.")
            return
        version_id = ver.get("version_id", "")
        try:
            result = self.client.restore_project(version_id=version_id)
            payload = result.get("project_payload")
            if not payload:
                QtWidgets.QMessageBox.warning(self, "No Payload", "This version has no project payload.")
                return
            import chisurf as cs
            from chisurf.core.project import Project as CSProject
            proj = CSProject.from_dict(payload)
            from chisurf.macros.core_fit import load_project_payload
            load_project_payload(proj, project_path=None)
            from qtpy import QtCore
            QtCore.QTimer.singleShot(0, lambda: self._restore_gui_from_fits(proj))
            if hasattr(cs, "cs") and cs.cs is not None:
                cs.cs._current_project_id = result.get("project_id")
                cs.cs._current_project_version_id = version_id
                cs.cs._current_project_name = result.get("project_name")
                cs.cs._current_project_visibility = result.get("visibility", "private")
            self.close()
        except Exception as exc:
            logging.error("Failed to restore project: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Restore Failed", str(exc))

    @staticmethod
    def _restore_gui_from_fits(proj: Any) -> None:
        try:
            from chisurf.macros.core_fit import restore_gui_from_fits
            fit_uids = [f.get("uid", f.get("uuid", "")) for f in (proj.fits or [])]
            if fit_uids:
                from qtpy import QtCore
                QtCore.QTimer.singleShot(0, lambda: restore_gui_from_fits(fit_uids))
        except Exception as exc:
            logging.error("Failed to restore fits: %s", exc)

    def _on_save(self) -> None:
        """Save the current project, prompting only for metadata when needed."""
        import chisurf as cs
        if not hasattr(cs, "cs") or cs.cs is None:
            QtWidgets.QMessageBox.warning(self, "No Project", "No project is currently open.")
            return

        current_project_id = getattr(cs.cs, "_current_project_id", None)
        current_version_id = getattr(cs.cs, "_current_project_version_id", None)
        project_name = getattr(cs.cs, "_current_project_name", "")
        visibility = getattr(cs.cs, "_current_project_visibility", "private")
        notes = None
        already_saved = bool(current_project_id or current_version_id)

        dlg = SaveProjectDialog(
            current_name=project_name,
            parent=self,
            allow_name_edit=not already_saved,
            title="Save New Version" if already_saved else "Save Project",
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        project_name = dlg.project_name
        visibility = dlg.visibility
        notes = dlg.notes

        if not project_name:
            QtWidgets.QMessageBox.warning(self, "No Project Name", "Project name is missing.")
            return

        from chisurf.macros.core_fit import get_project_payload
        try:
            payload = get_project_payload(project_name)
            payload_data = payload.to_dict() if hasattr(payload, "to_dict") else payload
            fit_count, dataset_count = self._project_counts_from_payload(payload_data)
            datasets = getattr(cs.cs, "imported_datasets", {}) or {}
            if not dataset_count and isinstance(datasets, dict):
                dataset_count = len(datasets)

            result = self.client.save_project(
                project_name=project_name,
                project_payload=payload_data,
                project_id=current_project_id,
                parent_version_id=current_version_id,
                notes=notes,
                visibility=visibility,
                fit_count=fit_count,
                dataset_count=dataset_count,
            )
            cs.cs._current_project_id = result.get("project_id")
            cs.cs._current_project_version_id = result.get("version_id")
            cs.cs._current_project_name = project_name
            cs.cs._current_project_visibility = result.get("visibility", visibility)
            logging.info(
                "Project saved: %s v%s (id=%s, ver=%s)",
                project_name, result.get("version_number"),
                result.get("project_id"), result.get("version_id"),
            )
            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"Project '{project_name}' saved as version {result.get('version_number')}.",
            )
            self.refresh()
        except Exception as exc:
            logging.error("Failed to save project: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Save Failed", str(exc))

    def _on_export(self) -> None:
        ver = self._selected_version()
        if not ver:
            QtWidgets.QMessageBox.information(self, "Select Version", "Please select a project version to export.")
            return
        version_id = ver.get("version_id", "")
        default_name = f"{ver.get('project_name', 'project')}_v{ver.get('version_number', 1)}.csp"
        target_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Project as .csp", default_name, "Chisurf Project (*.csp)",
        )
        if not target_path:
            return
        try:
            result = self.client.export_csp(version_id=version_id, target_path=target_path)
            if result.get("ok"):
                QtWidgets.QMessageBox.information(
                    self, "Exported",
                    f"Project exported to:\n{target_path}",
                )
            else:
                QtWidgets.QMessageBox.warning(self, "Export Failed", result.get("error", "Unknown error"))
        except Exception as exc:
            logging.error("Failed to export project: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Export Failed", str(exc))

    def _on_import(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Project", "", "Chisurf Project (*.csp)",
        )
        if not file_path:
            return
        try:
            preview = self.client.import_preview(file_path=file_path)
            if not preview.get("ok", True):
                QtWidgets.QMessageBox.warning(
                    self, "Preview Failed",
                    preview.get("error", "Unknown error"),
                )
                return
            collisions = preview.get("collisions", {})
            has_collisions = any(v for v in collisions.values())
            if has_collisions:
                dlg = CollisionDialog(collisions, preview.get("origin", {}), self)
                if dlg.exec() != QtWidgets.QDialog.Accepted:
                    return
            else:
                ok = QtWidgets.QMessageBox.question(
                    self, "Confirm Import",
                    f"No collisions detected.\n"
                    f"Original project: {preview.get('origin', {}).get('project_id', '?')}\n"
                    f"Contains: {preview.get('entity_counts', {}).get('operations', 0)} operations, "
                    f"{preview.get('entity_counts', {}).get('artifacts', 0)} artifacts, "
                    f"{preview.get('entity_counts', {}).get('objects', 0)} objects.\n\n"
                    "Proceed with import?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if ok != QtWidgets.QMessageBox.Yes:
                    return

            result = self.client.import_csp(
                file_path=file_path,
                resolve_collisions=has_collisions,
            )
            if result.get("ok"):
                QtWidgets.QMessageBox.information(
                    self, "Imported",
                    f"Project imported.\n"
                    f"New project ID: {result.get('project_id', '?')}\n"
                    f"Version: v{result.get('version_number', '?')}",
                )
                self.refresh()
            else:
                QtWidgets.QMessageBox.warning(self, "Import Failed", result.get("error", "Unknown error"))
        except Exception as exc:
            logging.error("Failed to import project: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Import Failed", str(exc))

    def _on_delete(self) -> None:
        ver = self._selected_version()
        if not ver:
            QtWidgets.QMessageBox.information(self, "Select Version", "Please select a project version to delete.")
            return
        version_id = ver.get("version_id", "")
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete",
            f"Delete version {ver.get('version_number', '?')} of project '{ver.get('project_name', '')}'?\n"
            f"ID: {version_id}\n\nThis action soft-deletes the version.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            result = self.client.delete_version(version_id=version_id)
            if result.get("ok"):
                self.refresh()
            else:
                QtWidgets.QMessageBox.warning(self, "Delete Failed", result.get("error", "Unknown error"))
        except Exception as exc:
            logging.error("Failed to delete version: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Delete Failed", str(exc))
