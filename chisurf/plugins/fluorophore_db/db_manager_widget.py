import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore, QtGui
from . import get_db

class SpectrumPlotPopup(QtWidgets.QDialog):
    def __init__(self, db, probe_id, chromophore_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Spectra: {chromophore_name}")
        self.resize(700, 500)
        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('bottom', 'Wavelength', units='nm')
        self.plot_widget.setLabel('left', 'Intensity', units='a.u.')
        self.layout.addWidget(self.plot_widget)
        
        with db:
            # Plot Absorption
            abs_spec = db.get_spectrum(probe_id, 'absorption')
            if abs_spec:
                wl, vals = abs_spec
                y = np.array(vals, dtype=float)
                if len(y) > 0 and np.max(y) > 0: y = y / np.max(y)
                self.plot_widget.plot(wl, y, pen=pg.mkPen('b', width=2), name="Absorption")
            
            # Plot Emission
            em_spec = db.get_spectrum(probe_id, 'emission')
            if em_spec:
                wl, vals = em_spec
                y = np.array(vals, dtype=float)
                if len(y) > 0 and np.max(y) > 0: y = y / np.max(y)
                self.plot_widget.plot(wl, y, pen=pg.mkPen('r', width=2), name="Emission")
            
            if not abs_spec and not em_spec:
                self.layout.addWidget(QtWidgets.QLabel("No spectra data found."))

class FluorophoreDBWidget(QtWidgets.QMainWindow):
    """Main curation interface for the Fluorophore Database."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fluorophore Database Manager")
        self.resize(1000, 600)
        
        self.db = get_db()
        
        self._setup_ui()
        self.refresh_table()

    def _setup_ui(self):
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # Filter Area
        self.filter_layout = QtWidgets.QHBoxLayout()
        self.filter_layout.addWidget(QtWidgets.QLabel("Name Filter:"))
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Search fluorophores...")
        self.filter_edit.textChanged.connect(self.apply_filter)
        self.filter_layout.addWidget(self.filter_edit)
        self.layout.addLayout(self.filter_layout)
        # Toolbar
        self.toolbar = QtWidgets.QToolBar()
        self.addToolBar(self.toolbar)
        
        self.add_action = QtWidgets.QAction("\U00002795 Add New...", self)
        self.add_action.triggered.connect(self.on_add_item)
        self.toolbar.addAction(self.add_action)
        
        self.edit_action = QtWidgets.QAction("\U0000270F\U0000FE0F Edit Selected...", self)
        self.edit_action.triggered.connect(self.on_edit_item)
        self.toolbar.addAction(self.edit_action)
        
        self.delete_action = QtWidgets.QAction("\U0001F5D1\U0000FE0F Delete Selected", self)
        self.delete_action.triggered.connect(self.on_delete_item)
        self.toolbar.addAction(self.delete_action)
        
        self.toolbar.addSeparator()
        
        self.mark_good_action = QtWidgets.QAction("\U00002B50 Toggle Curated", self)
        self.mark_good_action.setToolTip("Toggle Curated status (Shortcut: Space)")
        self.mark_good_action.triggered.connect(lambda: self.toggle_selection_curated())
        self.toolbar.addAction(self.mark_good_action)
        
        self.mark_quality_action = QtWidgets.QAction("\U00002705 Toggle Quality", self)
        self.mark_quality_action.setToolTip("Toggle Quality Good/Bad (Shortcut: Q)")
        self.mark_quality_action.triggered.connect(lambda: self.toggle_selection_quality())
        self.toolbar.addAction(self.mark_quality_action)
        
        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Category", "Source", "Curated", "Quality", "Abs Max", "Em Max", "QY", "Lifetime", "Extinction"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.itemDoubleClicked.connect(self.on_edit_item)
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.itemChanged.connect(self.on_item_changed)
        self.layout.addWidget(self.table)
        
        # Status
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        with self.db:
            items = self.db.get_standardized_items(include_uncurated=True)
            types = self.db.get_probe_types_dict()
            
            for i, item in enumerate(items):
                self.table.insertRow(i)
                
                # ID
                id_item = QtWidgets.QTableWidgetItem(str(item['probe_id']))
                id_item.setFlags(id_item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, 0, id_item)
                
                # Name
                name_item = QtWidgets.QTableWidgetItem(item['chromophore_name'] or "")
                self.table.setItem(i, 1, name_item)
                
                # Category
                cat_item = QtWidgets.QTableWidgetItem(item['category'] or "")
                self.table.setItem(i, 2, cat_item)
                
                # Source (type)
                type_display = types.get(item['type_id'], "Unknown")
                type_item = QtWidgets.QTableWidgetItem(type_display)
                type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, 3, type_item)
                
                # Curated
                is_curated = bool(item['is_curated'])
                curated_item = QtWidgets.QTableWidgetItem("✓" if is_curated else "✗")
                curated_item.setTextAlignment(QtCore.Qt.AlignCenter)
                curated_item.setFlags(curated_item.flags() & ~QtCore.Qt.ItemIsEditable)
                if not is_curated:
                    curated_item.setForeground(QtGui.QColor("red"))
                self.table.setItem(i, 4, curated_item)
                
                # Quality
                is_good_quality = bool(item['quality_flag'])
                quality_item = QtWidgets.QTableWidgetItem("Good" if is_good_quality else "Bad")
                quality_item.setTextAlignment(QtCore.Qt.AlignCenter)
                quality_item.setFlags(quality_item.flags() & ~QtCore.Qt.ItemIsEditable)
                if not is_good_quality:
                    quality_item.setForeground(QtGui.QColor("orange"))
                self.table.setItem(i, 5, quality_item)
                
                # Standardized Optical properties (formatted for display)
                abs_max = self.format_value(item.get("abs_max"))
                em_max = self.format_value(item.get("em_max"))
                qy = self.format_value(item.get("qy"), is_qy=True)
                lifetime = self.format_value(item.get("lifetime"))
                ext = self.format_value(item.get("ext_coeff"))
                
                self.table.setItem(i, 6, QtWidgets.QTableWidgetItem(abs_max))
                self.table.setItem(i, 7, QtWidgets.QTableWidgetItem(em_max))
                self.table.setItem(i, 8, QtWidgets.QTableWidgetItem(qy))
                self.table.setItem(i, 9, QtWidgets.QTableWidgetItem(lifetime))
                self.table.setItem(i, 10, QtWidgets.QTableWidgetItem(ext))

        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.apply_filter() 
        self.status.showMessage(f"Loaded {self.table.rowCount()} items.")

    def on_item_changed(self, item):
        row = item.row()
        col = item.column()
        
        try:
            item_id = int(self.table.item(row, 0).text())
        except (AttributeError, ValueError): return
        
        new_val = item.text().strip()
        
        with self.db:
            db_item = self.db.get_probe_by_id(item_id)
            if not db_item: return
            
            if col == 1: # Name
                self.db.update_probe(item_id, chromophore_name=new_val)
                self.status.showMessage(f"Updated name for item {item_id}: {new_val}")
            
            elif col == 2: # Category
                self.db.update_probe(item_id, category=new_val)
                self.status.showMessage(f"Updated category for item {item_id}: {new_val}")

            elif col in (6, 7, 8, 9, 10): # Optical Properties
                prop_map = {
                    6: "λabs",
                    7: "λfl",
                    8: "ηfl",
                    9: "τfl",
                    10: "εmax"
                }
                prop_name = prop_map.get(col)
                if prop_name:
                    self.db.add_optical_property(item_id, prop_name, new_val)
                    self.status.showMessage(f"Updated {prop_name} for item {item_id}: {new_val}")

    def apply_filter(self):
        text = self.filter_edit.text().lower()
        for i in range(self.table.rowCount()):
            name = self.table.item(i, 1).text().lower()
            self.table.setRowHidden(i, text not in name)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.toggle_selection_curated()
        elif event.key() == QtCore.Qt.Key_Q:
            self.toggle_selection_quality()
        else:
            super().keyPressEvent(event)

    def on_cell_clicked(self, row, col):
        # Peak Absorption (Col 6) or Emission (Col 7) in new layout
        if col in (6, 7):
            probe_id = int(self.table.item(row, 0).text())
            chromophore_name = self.table.item(row, 1).text()
            dlg = SpectrumPlotPopup(self.db, probe_id, chromophore_name, self)
            dlg.exec_()

    def format_value(self, val, is_qy=False):
        if val is None: return ""
        s = str(val).strip()
        try:
            v = float(s)
            if v >= 1000 or (v > 0 and v < 0.1):
                return f"{v:.2e}"
            else:
                return f"{v:g}"
        except:
            return s


    def on_add_item(self):
        dlg = AddFluorophoreDialog(self.db, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.refresh_table()

    def on_edit_item(self):
        row = self.table.currentRow()
        if row < 0:
            return
        item_id = int(self.table.item(row, 0).text())
        
        from .editor import MetadataEditorDialog
        dlg = MetadataEditorDialog(self.db, item_id, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.refresh_table()

    def on_delete_item(self):
        rows = sorted(set(index.row() for index in self.table.selectedIndexes()), reverse=True)
        if not rows:
            return
            
        res = QtWidgets.QMessageBox.question(self, "Confirm Delete", f"Delete {len(rows)} selected items?")
        if res == QtWidgets.QMessageBox.Yes:
            with self.db:
                for row in rows:
                    probe_id = int(self.table.item(row, 0).text())
                    self.db.delete_probe(probe_id)
            self.refresh_table()

    def toggle_selection_curated(self):
        rows = set(index.row() for index in self.table.selectedIndexes())
        if not rows: return
        with self.db:
            for row in rows:
                probe_id = int(self.table.item(row, 0).text())
                item = self.db.get_probe_by_id(probe_id)
                if item:
                    self.db.update_probe(probe_id, is_curated=not bool(item['is_curated']))
        self.refresh_table()

    def toggle_selection_quality(self):
        rows = set(index.row() for index in self.table.selectedIndexes())
        if not rows: return
        with self.db:
            for row in rows:
                probe_id = int(self.table.item(row, 0).text())
                item = self.db.get_probe_by_id(probe_id)
                if item:
                    self.db.update_probe(probe_id, quality_flag=not bool(item['quality_flag']))
        self.refresh_table()

class AddFluorophoreDialog(QtWidgets.QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Add New Fluorophore")
        self.resize(400, 300)
        
        self.layout = QtWidgets.QFormLayout(self)
        
        self.name_edit = QtWidgets.QLineEdit()
        self.layout.addRow("Name:", self.name_edit)
        
        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.addItems(["Organic Dyes", "Fluorescent Proteins", "Detectors", "Filters", "Dichroics", "Lenses", "Other"])
        self.layout.addRow("Category:", self.category_combo)

        self.type_combo = QtWidgets.QComboBox()
        with self.db:
            for r in self.db.get_probe_types():
                self.type_combo.addItem(r['display_name'], r['type_id'])
        self.layout.addRow("Source:", self.type_combo)
        
        self.quality_check = QtWidgets.QCheckBox("Good Quality")
        self.quality_check.setChecked(True)
        self.layout.addRow("Quality:", self.quality_check)

        self.desc_edit = QtWidgets.QPlainTextEdit()
        self.layout.addRow("Description:", self.desc_edit)
        
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept_data)
        self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)

    def accept_data(self):
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Error", "Name cannot be empty.")
            return
            
        type_id = self.type_combo.currentData()
        category = self.category_combo.currentText().strip()
        quality = self.quality_check.isChecked()
        desc = self.desc_edit.toPlainText().strip()
        
        try:
            with self.db:
                probe_id = self.db.add_probe(name, type_id, desc)
                self.db.update_probe(probe_id, category=category, quality_flag=1 if quality else 0)
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to add item:\n{e}")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = FluorophoreDBWidget()
    w.show()
    sys.exit(app.exec_())
