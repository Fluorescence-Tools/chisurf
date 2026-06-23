import json
from qtpy import QtWidgets, QtCore, QtGui

class MetadataEditorDialog(QtWidgets.QDialog):
    def __init__(self, db, probe_id, parent=None):
        super().__init__(parent)
        self.db = db
        self.probe_id = probe_id
        self.setWindowTitle("Edit Metadata")
        self.resize(600, 500)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.info_label = QtWidgets.QLabel()
        self.layout.addWidget(self.info_label)
        
        self.editor = QtWidgets.QPlainTextEdit()
        font = QtGui.QFont("Courier" if QtCore.QSysInfo.productType() == "windows" else "Monospace")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.editor.setFont(font)
        self.layout.addWidget(self.editor)
        
        help_text = QtWidgets.QLabel("Edit the JSON to modify properties. 'is_curated' applies to 'is_good' in DB.")
        help_text.setStyleSheet("color: gray; font-style: italic;")
        self.layout.addWidget(help_text)
        
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.save_data)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        
        self.load_data()
        
    def load_data(self):
        with self.db:
            item = self.db.get_probe_by_id(self.probe_id)
            if not item:
                self.editor.setPlainText("{}")
                return
            
            # Get type name
            type_name = "Unknown"
            for r in self.db.get_probe_types():
                if r['id'] == item['type_id']:
                    type_name = r['display_name']
                    break
                    
            self.info_label.setText(f"<b>Source:</b> {type_name} <br/> <b>ID:</b> {item['probe_id']}")
            
            opt_props = self.db.get_optical_properties(self.probe_id)
            
            data = {
                "chromophore_name": item['chromophore_name'],
                "description": item['description'] or "",
                "category": item['category'] or "",
                "quality_flag": bool(item['quality_flag']),
                "is_curated": bool(item['is_curated']),
                "probe_origin": item['probe_origin'],
                "probe_link_type": item['probe_link_type'],
                "fluorophore_type": item['fluorophore_type'],
                "optical_properties": opt_props
            }
            
            self.editor.setPlainText(json.dumps(data, indent=4))
            
    def save_data(self):
        text = self.editor.toPlainText()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            QtWidgets.QMessageBox.critical(self, "JSON Error", f"Invalid JSON:\n{e}")
            return
            
        name = data.get("name", "")
        description = data.get("description", "")
        category = data.get("category", "")
        quality = bool(data.get("quality_is_good", True))
        is_good = bool(data.get("is_curated", True))
        opt_props = data.get("optical_properties", {})
        
        if not isinstance(opt_props, dict):
            QtWidgets.QMessageBox.critical(self, "Data Error", "'optical_properties' must be a JSON object (dictionary).")
            return
        
        try:
            with self.db:
                # Standard fields
                update_fields = {
                    "chromophore_name": data.get("chromophore_name", data.get("name", "")),
                    "description": data.get("description", ""),
                    "category": data.get("category", ""),
                    "quality_flag": 1 if data.get("quality_flag", data.get("quality_is_good", True)) else 0,
                    "is_curated": 1 if data.get("is_curated", True) else 0,
                    "probe_origin": data.get("probe_origin", "extrinsic"),
                    "probe_link_type": data.get("probe_link_type", "covalent"),
                    "fluorophore_type": data.get("fluorophore_type", "unspecified")
                }
                self.db.update_probe(self.probe_id, **update_fields)
                
                # Optical properties
                self.db.clear_optical_properties(self.probe_id)
                opt_props = data.get("optical_properties", {})
                if isinstance(opt_props, dict):
                    for k, v in opt_props.items():
                        self.db.add_optical_property(self.probe_id, str(k), str(v))
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Database Error", f"Failed to save:\n{e}")
