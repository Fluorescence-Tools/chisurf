"""Qt widgets for selecting and creating MFDB samples (PRD-39 GUI)."""
from __future__ import annotations

from qtpy import QtCore, QtWidgets

from chisurf.core.mfdb.samples.external_refs import diff_sequences, fetch_uniprot
from chisurf.core.mfdb.models import (
    EntityDefinition,
    MutationDefinition,
    SampleDefinition,
)
from chisurf.core.mfdb.samples.sample_manager import create_sample, list_samples


class SamplePicker(QtWidgets.QWidget):
    """Combo box with a ``New...`` option for selecting a sample.

    Signals
    -------
    sample_changed : Signal(str)
        Emitted when the selected sample ID changes.
    """

    sample_changed = QtCore.Signal(str)

    def __init__(self, db=None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._samples: list[dict[str, str]] = []
        self._updating = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Sample:"))
        self._combo = QtWidgets.QComboBox(self)
        self._combo.setMinimumWidth(220)
        self._combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self._combo, 1)

        self._new_btn = QtWidgets.QPushButton("New...", self)
        self._new_btn.clicked.connect(self._on_new_sample)
        layout.addWidget(self._new_btn)

        self.refresh()

    def set_db(self, db) -> None:
        self._db = db
        self.refresh()

    def set_samples(self, samples: list[dict[str, str]]) -> None:
        self._samples = samples
        self._updating = True
        try:
            self._combo.blockSignals(True)
            self._combo.clear()
            self._combo.addItem("(no sample)", "")
            for sample in samples:
                sample_id = sample.get("sample_id") or sample.get("id") or ""
                name = sample.get("name") or sample.get("display_name") or sample_id
                self._combo.addItem(str(name), str(sample_id))
        finally:
            self._combo.blockSignals(False)
            self._updating = False

    def refresh(self) -> None:
        if self._db is None:
            self.set_samples([])
            return
        try:
            self.set_samples(list_samples(self._db))
        except Exception:
            self.set_samples([])

    def selected_sample_id(self) -> str:
        return self._combo.currentData() or ""

    def set_selected(self, sample_id: str) -> None:
        for idx in range(self._combo.count()):
            if self._combo.itemData(idx) == sample_id:
                self._combo.setCurrentIndex(idx)
                return

    def _on_selection_changed(self, index: int) -> None:
        if self._updating:
            return
        sample_id = self._combo.itemData(index) or ""
        self.sample_changed.emit(sample_id)

    def _on_new_sample(self) -> None:
        dialog = _SampleDefinitionDialog(self._db, parent=self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        definition = dialog.definition
        if definition is None:
            return
        if self._db is not None:
            try:
                sample_id = create_sample(self._db, definition)
                self.refresh()
                self.set_selected(sample_id)
                self.sample_changed.emit(sample_id)
                return
            except Exception:
                pass
        self.sample_changed.emit(f"__new__:{definition.name.strip()}")


class _SampleDefinitionDialog(QtWidgets.QDialog):
    """Dialog for entering sample definition with PRD-39 external references."""

    def __init__(self, db=None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self.definition: SampleDefinition | None = None
        self.setWindowTitle("New Sample")
        self.resize(520, 620)

        main_layout = QtWidgets.QVBoxLayout(self)

        # ── Basic sample fields ──
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit(self)
        self.description_edit = QtWidgets.QLineEdit(self)
        self.entity_name_edit = QtWidgets.QLineEdit(self)
        self.entity_sequence_edit = QtWidgets.QLineEdit(self)
        self.entity_type_edit = QtWidgets.QLineEdit(self)
        self.donor_edit = QtWidgets.QLineEdit(self)
        self.acceptor_edit = QtWidgets.QLineEdit(self)
        self.buffer_edit = QtWidgets.QLineEdit(self)

        form.addRow("Sample name", self.name_edit)
        form.addRow("Description", self.description_edit)
        form.addRow("Entity name", self.entity_name_edit)
        form.addRow("Entity sequence", self.entity_sequence_edit)
        form.addRow("Entity type", self.entity_type_edit)
        form.addRow("Donor probe", self.donor_edit)
        form.addRow("Acceptor probe", self.acceptor_edit)
        form.addRow("Buffer", self.buffer_edit)
        main_layout.addLayout(form)

        # ── External references group (PRD-39) ──
        ref_group = QtWidgets.QGroupBox("External References (UniProt / PDB)", self)
        ref_layout = QtWidgets.QFormLayout(ref_group)

        self.uniprot_edit = QtWidgets.QLineEdit(self)
        self.uniprot_edit.setPlaceholderText("e.g. P00720")
        fetch_btn = QtWidgets.QPushButton("Fetch", self)
        fetch_btn.clicked.connect(self._on_fetch_uniprot)
        uniprot_row = QtWidgets.QHBoxLayout()
        uniprot_row.addWidget(self.uniprot_edit, 1)
        uniprot_row.addWidget(fetch_btn)
        ref_layout.addRow("UniProt accession", uniprot_row)

        self.pdb_id_edit = QtWidgets.QLineEdit(self)
        self.pdb_id_edit.setPlaceholderText("e.g. 2LZM")
        ref_layout.addRow("PDB ID", self.pdb_id_edit)

        self.pdb_chain_edit = QtWidgets.QLineEdit(self)
        self.pdb_chain_edit.setPlaceholderText("e.g. A")
        ref_layout.addRow("PDB chain ID", self.pdb_chain_edit)

        self.organism_edit = QtWidgets.QLineEdit(self)
        self.organism_edit.setPlaceholderText("Auto-populated from UniProt")
        ref_layout.addRow("Organism", self.organism_edit)

        self.ref_sequence_edit = QtWidgets.QLineEdit(self)
        self.ref_sequence_edit.setPlaceholderText("Canonical / WT sequence from UniProt")
        ref_layout.addRow("Reference sequence", self.ref_sequence_edit)

        diff_btn = QtWidgets.QPushButton("Diff vs Reference", self)
        diff_btn.clicked.connect(self._on_diff_vs_reference)
        ref_layout.addRow(diff_btn)

        main_layout.addWidget(ref_group)

        # ── Mutations table ──
        mut_group = QtWidgets.QGroupBox("Mutations (struct_ref_seq_dif)", self)
        mut_layout = QtWidgets.QVBoxLayout(mut_group)

        self._mut_table = QtWidgets.QTableWidget(0, 5, self)
        self._mut_table.setHorizontalHeaderLabels(
            ["seq_id", "mut_comp_id", "wt_comp_id", "auth_name", "kind"]
        )
        self._mut_table.horizontalHeader().setStretchLastSection(True)
        self._mut_table.setAlternatingRowColors(True)
        mut_layout.addWidget(self._mut_table)

        mut_btn_row = QtWidgets.QHBoxLayout()
        add_mut_btn = QtWidgets.QPushButton("Add mutation", self)
        add_mut_btn.clicked.connect(self._on_add_mutation)
        clear_mut_btn = QtWidgets.QPushButton("Clear", self)
        clear_mut_btn.clicked.connect(self._on_clear_mutations)
        mut_btn_row.addWidget(add_mut_btn)
        mut_btn_row.addWidget(clear_mut_btn)
        mut_btn_row.addStretch()
        mut_layout.addLayout(mut_btn_row)

        main_layout.addWidget(mut_group)

        # ── Buttons ──
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    # ── UniProt fetch (offline-safe, non-blocking) ──────────────────────

    def _on_fetch_uniprot(self) -> None:
        accession = self.uniprot_edit.text().strip()
        if not accession:
            QtWidgets.QMessageBox.information(self, "Fetch", "Enter a UniProt accession first.")
            return
        cache_dir = None
        if self._db is not None:
            import pathlib
            cache_dir = pathlib.Path(self._db.db_path).parent / "uniprot_cache"
        result = fetch_uniprot(accession, cache_dir=cache_dir)
        if result is None:
            QtWidgets.QMessageBox.warning(
                self, "Fetch failed",
                f"Could not fetch UniProt entry {accession}. "
                "Check the accession and network connectivity.",
            )
            return
        if result.get("organism"):
            self.organism_edit.setText(result["organism"])
        if result.get("sequence"):
            self.ref_sequence_edit.setText(result["sequence"])

    # ── Diff vs reference ──────────────────────────────────────────────

    def _on_diff_vs_reference(self) -> None:
        construct = self.entity_sequence_edit.text().strip()
        reference = self.ref_sequence_edit.text().strip()
        if not construct:
            QtWidgets.QMessageBox.information(
                self, "Diff", "Enter an entity sequence first."
            )
            return
        if not reference:
            QtWidgets.QMessageBox.information(
                self, "Diff", "Enter a reference sequence first (or click Fetch)."
            )
            return
        try:
            mutations = diff_sequences(construct, reference)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Diff failed", str(exc))
            return

        if not mutations:
            QtWidgets.QMessageBox.information(
                self, "Diff", "No differences found — construct matches reference."
            )
            return

        self._populate_mutation_table(mutations)

        count = len(mutations)
        QtWidgets.QMessageBox.information(
            self, "Diff complete",
            f"Found {count} mutation{'s' if count != 1 else ''}. "
            "Review the mutations table and edit if needed.",
        )

    # ── Mutation table helpers ─────────────────────────────────────────

    def _populate_mutation_table(self, mutations: list[MutationDefinition]) -> None:
        self._mut_table.setRowCount(len(mutations))
        for row, mut in enumerate(mutations):
            self._mut_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(mut.seq_id)))
            self._mut_table.setItem(row, 1, QtWidgets.QTableWidgetItem(mut.mut_comp_id))
            self._mut_table.setItem(row, 2, QtWidgets.QTableWidgetItem(mut.wt_comp_id))
            self._mut_table.setItem(row, 3, QtWidgets.QTableWidgetItem(mut.auth_name))
            self._mut_table.setItem(row, 4, QtWidgets.QTableWidgetItem(mut.kind))
        self._mut_table.resizeColumnsToContents()

    def _on_add_mutation(self) -> None:
        row = self._mut_table.rowCount()
        self._mut_table.insertRow(row)
        self._mut_table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self._mut_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        self._mut_table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
        self._mut_table.setItem(row, 3, QtWidgets.QTableWidgetItem(""))
        self._mut_table.setItem(row, 4, QtWidgets.QTableWidgetItem("engineered_mutation"))

    def _on_clear_mutations(self) -> None:
        self._mut_table.setRowCount(0)

    def _read_mutations_from_table(self) -> list[MutationDefinition]:
        mutations: list[MutationDefinition] = []
        for row in range(self._mut_table.rowCount()):
            seq_id_item = self._mut_table.item(row, 0)
            if seq_id_item is None or not seq_id_item.text().strip():
                continue
            try:
                seq_id = int(seq_id_item.text().strip())
            except ValueError:
                continue
            mut_comp_id = (self._mut_table.item(row, 1).text().strip()
                           if self._mut_table.item(row, 1) else "")
            wt_comp_id = (self._mut_table.item(row, 2).text().strip()
                          if self._mut_table.item(row, 2) else "")
            auth_name = (self._mut_table.item(row, 3).text().strip()
                         if self._mut_table.item(row, 3) else "")
            kind = (self._mut_table.item(row, 4).text().strip()
                    if self._mut_table.item(row, 4) else "engineered_mutation")
            mutations.append(MutationDefinition(
                seq_id=seq_id,
                mut_comp_id=mut_comp_id,
                wt_comp_id=wt_comp_id,
                auth_name=auth_name,
                kind=kind,
            ))
        return mutations

    # ── Accept ─────────────────────────────────────────────────────────

    def accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Missing name", "Sample name is required.")
            return

        entity = EntityDefinition(
            name=self.entity_name_edit.text().strip() or name,
            entity_type=self.entity_type_edit.text().strip() or "protein",
            sequence=self.entity_sequence_edit.text().strip() or "",
            uniprot_accession=self.uniprot_edit.text().strip() or None,
            pdb_id=self.pdb_id_edit.text().strip() or None,
            pdb_chain_id=self.pdb_chain_edit.text().strip() or None,
            organism=self.organism_edit.text().strip() or None,
            reference_sequence=self.ref_sequence_edit.text().strip() or None,
            mutations=self._read_mutations_from_table(),
        )

        self.definition = SampleDefinition(
            name=name,
            description=self.description_edit.text().strip(),
            entities=[entity],
            entity_name=self.entity_name_edit.text().strip(),
            entity_sequence=self.entity_sequence_edit.text().strip(),
            entity_type=self.entity_type_edit.text().strip(),
            donor_probe_name=self.donor_edit.text().strip(),
            acceptor_probe_name=self.acceptor_edit.text().strip(),
            buffer_description=self.buffer_edit.text().strip(),
        )
        super().accept()


def show_sample_picker_dialog(db=None, parent: QtWidgets.QWidget | None = None) -> str | None:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Select sample")
    layout = QtWidgets.QVBoxLayout(dialog)
    picker = SamplePicker(db=db, parent=dialog)
    layout.addWidget(picker)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
        dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    skip = QtWidgets.QPushButton("Skip", dialog)
    buttons.addButton(skip, QtWidgets.QDialogButtonBox.ActionRole)
    skip.clicked.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return picker.selected_sample_id() or None
