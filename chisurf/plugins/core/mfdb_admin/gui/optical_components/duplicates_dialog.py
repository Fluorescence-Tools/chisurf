from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtWidgets

import re
import math
from collections import defaultdict
from difflib import SequenceMatcher

def _norm(name: str) -> str:
    n = (name or "").lower()
    for filler in ["fluor", "dye", "fluorescent"]:
        n = n.replace(filler, "")
    if n.startswith("af-") or n.startswith("af "):
        n = n.replace("af", "alexa", 1)
    elif n.startswith("af") and len(n) > 2 and n[2].isdigit():
        n = "alexa" + n[2:]
    n = n.replace("cyanine", "cy")
    return re.sub(r'[^a-z0-9]', '', n)

def _calc_prob(p1: dict, p2: dict) -> float:
    # Prior probability
    P_dup = 0.001
    P_not_dup = 1 - P_dup
    
    # 1. Name Match
    n1 = _norm(p1["chromophore_name"])
    n2 = _norm(p2["chromophore_name"])
    
    if n1 == n2:
        L_name_dup = 0.99
        L_name_not = 0.0001
    else:
        sim = SequenceMatcher(None, n1, n2).ratio()
                
        if sim < 0.6:
            L_name_dup = 1e-6
            L_name_not = 0.99
        else:
            L_name_dup = math.exp(-10 * (1 - sim)**2) 
            L_name_not = 0.01 if sim > 0.8 else 0.1
        
    # 2. Abs Max Match
    a1, a2 = p1.get("abs_max"), p2.get("abs_max")
    if a1 and a2:
        try:
            diff = abs(float(a1) - float(a2))
            L_abs_dup = math.exp(-0.5 * (diff / 5.0)**2)
            L_abs_not = 1.0 / 300.0  # Approx chance of random match in 300nm range
        except ValueError:
            L_abs_dup = L_abs_not = 1.0
    else:
        # If missing, it tells us nothing
        L_abs_dup = L_abs_not = 1.0
        
    # 3. Em Max Match
    e1, e2 = p1.get("em_max"), p2.get("em_max")
    if e1 and e2:
        try:
            diff = abs(float(e1) - float(e2))
            L_em_dup = math.exp(-0.5 * (diff / 5.0)**2)
            L_em_not = 1.0 / 300.0
        except ValueError:
            L_em_dup = L_em_not = 1.0
    else:
        L_em_dup = L_em_not = 1.0
        
    num = P_dup * L_name_dup * L_abs_dup * L_em_dup
    den = num + P_not_dup * L_name_not * L_abs_not * L_em_not
    prob = num / den if den > 0 else 0
    
    # Strongly penalize duplicates from the same curated source
    s1, s2 = p1.get("source"), p2.get("source")
    if s1 and s2 and s1 == s2 and s1 in ("fpbase", "pubmed"):
        prob *= 0.01  # Highly unlikely that fpbase has exact duplicates of its own entries
        
    return prob

class DuplicateFinderThread(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished_groups = QtCore.Signal(list)
    
    def __init__(self, probes: list[dict[str, Any]], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.probes = probes
        
    def run(self) -> None:
        by_category = defaultdict(list)
        for p in self.probes:
            by_category[p.get("category", "other")].append(p)
            
        edges = []
        total_categories = len(by_category)
        
        for cat_idx, cat_probes in enumerate(by_category.values()):
            if self.isInterruptionRequested():
                return
                
            self.progress.emit(int(cat_idx / max(1, total_categories) * 90))
            
            # Sort probes by abs_max (or a large number if None)
            def _get_abs(p):
                try:
                    return float(p.get("abs_max"))
                except (TypeError, ValueError):
                    return 999999.0
                    
            cat_probes.sort(key=_get_abs)
            
            n = len(cat_probes)
            for i in range(n):
                if self.isInterruptionRequested():
                    return
                    
                p1 = cat_probes[i]
                a1 = _get_abs(p1)
                
                for j in range(i + 1, n):
                    p2 = cat_probes[j]
                    a2 = _get_abs(p2)
                    
                    # Exact name matches are always checked
                    if _norm(p1["chromophore_name"]) == _norm(p2["chromophore_name"]):
                        edges.append((p1["probe_id"], p2["probe_id"]))
                        continue
                    
                    # If abs_max diff > 15nm, break inner loop (since sorted)
                    if a1 != 999999.0 and a2 != 999999.0:
                        if a2 - a1 > 15.0:
                            break
                            
                    # Otherwise compute full probability
                    prob = _calc_prob(p1, p2)
                    if prob > 0.8:
                        edges.append((p1["probe_id"], p2["probe_id"]))
                        
        self.progress.emit(95)
        
        # Connected components
        parent = {}
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        for p in self.probes:
            parent[p["probe_id"]] = p["probe_id"]
            
        for i, j in edges:
            union(i, j)
            
        groups_by_root = defaultdict(list)
        probe_map = {p["probe_id"]: p for p in self.probes}
        
        for pid in parent:
            root = find(pid)
            groups_by_root[root].append(probe_map[pid])
            
        duplicate_groups = []
        for root, group_probes in groups_by_root.items():
            if len(group_probes) > 1:
                group_probes.sort(key=lambda p: len(p["chromophore_name"]))
                group_name = group_probes[0]["chromophore_name"]
                duplicate_groups.append({
                    "norm_name": group_name,
                    "probes": group_probes
                })
                
        duplicate_groups.sort(key=lambda g: g["norm_name"].lower())
        self.progress.emit(100)
        self.finished_groups.emit(duplicate_groups)


class GroupDetailModel:
    def __init__(self, group_name: str, probes: list[dict[str, Any]], spectra_data: list[dict], selected_probe_id: int | None = None):
        self.group_name = group_name
        self.probes = probes
        self.spectra_data = spectra_data
        self.selected_probe_id = selected_probe_id
        
        self.spec_types = sorted(list({sp["spectrum_type"] for sp in self.spectra_data if sp["wavelengths"] and sp["intensity_values"]}))
        
        # dynamic attrs for metadata panels
        for p in self.probes:
            pid = p["probe_id"]
            setattr(self, f"name_{pid}", p.get("chromophore_name", ""))
            setattr(self, f"category_{pid}", p.get("category", ""))
            setattr(self, f"source_{pid}", p.get("source", ""))
            
            for prop_name, prop_val in p.get("optical_properties", {}).items():
                setattr(self, f"op_{pid}_{prop_name}", str(prop_val))
                
        self._plot_data = {}
        colors = ["y", "c", "m", "g", "r", "w", (150, 150, 255), (255, 150, 150), (150, 255, 150)]
        for idx, st in enumerate(self.spec_types):
            series = []
            for sp in self.spectra_data:
                if sp["spectrum_type"] == st and sp["wavelengths"] and sp["intensity_values"]:
                    pid = sp["probe_id"]
                    probe = next((p for p in self.probes if p["probe_id"] == pid), None)
                    name = f"ID {pid} - {probe.get('chromophore_name')}" if probe else f"ID {pid}"
                    
                    probe_idx = self.probes.index(probe) if probe in self.probes else 0
                    color = colors[probe_idx % len(colors)]
                    
                    if self.selected_probe_id is None:
                        width = 2
                        style = "solid"
                    elif pid == self.selected_probe_id:
                        width = 3
                        style = "solid"
                    else:
                        width = 1
                        style = "dash"
                        color = (150, 150, 150) # make unselected grey
                        
                    series.append({
                        "x": sp["wavelengths"],
                        "y": sp["intensity_values"],
                        "name": name,
                        "color": color,
                        "width": width,
                        "style": style
                    })
            
            method_name = f"get_plot_data_{idx}"
            self._plot_data[method_name] = series
            setattr(self, method_name, lambda k=method_name: self._plot_data[k])

    def plot_view_spec(self):
        from chisurf.core.dataspec import ModelView, PlotSection, PanelSection
        sections = []
        spec_sections = []
        for idx, st in enumerate(self.spec_types):
            spec_sections.append(PlotSection(
                source=f"get_plot_data_{idx}",
                x_label="Wavelength",
                y_label="Intensity",
                height=250,
                title=f"{st.title()} Spectrum"
            ))
            
        if spec_sections:
            sections.append(PanelSection(sections=tuple(spec_sections), title="Spectra Comparison"))
        return ModelView(sections=tuple(sections))

    def meta_view_spec(self):
        from chisurf.core.dataspec import ModelView, PanelSection, ValueSection
        sections = []
        for p in self.probes:
            pid = p["probe_id"]
            probe_sections = []
            probe_sections.append(ValueSection(label="Name", attr=f"name_{pid}", kind="str"))
            probe_sections.append(ValueSection(label="Category", attr=f"category_{pid}", kind="str"))
            probe_sections.append(ValueSection(label="Source", attr=f"source_{pid}", kind="str"))
            
            for prop_name, prop_val in p.get("optical_properties", {}).items():
                probe_sections.append(ValueSection(label=prop_name.replace("_", " ").title(), attr=f"op_{pid}_{prop_name}", kind="str"))
                
            sections.append(PanelSection(
                title=f"Metadata: ID {pid} - {p.get('chromophore_name')}",
                sections=tuple(probe_sections),
                collapsed=True
            ))
        return ModelView(sections=tuple(sections))


class DuplicatesDialog(QtWidgets.QDialog):
    """Dialog for reviewing potential duplicate probes and merging them."""

    def __init__(self, duplicate_groups: list[dict[str, Any]], client: Any, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Find and Merge Duplicates")
        self.resize(1000, 600)
        
        self._duplicate_groups = duplicate_groups
        self._client = client
        self._primary_selections: dict[str, int] = {}  # norm_name -> primary_probe_id
        self._spectra_cache: dict[str, list[dict]] = {} # norm_name -> spectra_data
        
        self._setup_ui()
        self._populate()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        
        info_label = QtWidgets.QLabel(
            "Potential duplicates identified by name similarity.\n"
            "Select the primary probe for each group in the tree below. "
            "Use the panel on the right to compare spectra for the selected group.\n"
            "When you merge, the primary probe keeps its spectra and missing ones are filled from duplicates. Overlaps are ignored."
        )
        info_label.setWordWrap(True)
        info_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        layout.addWidget(info_label)
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Primary", "ID", "Name", "Category", "Source", "Status"])
        self._tree.currentItemChanged.connect(self._on_tree_selection)
        splitter.addWidget(self._tree)
        
        self._inner_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        self._plot_scroll = QtWidgets.QScrollArea()
        self._plot_scroll.setWidgetResizable(True)
        self._plot_widget = QtWidgets.QWidget()
        self._plot_scroll.setWidget(self._plot_widget)
        self._inner_splitter.addWidget(self._plot_scroll)
        
        self._meta_scroll = QtWidgets.QScrollArea()
        self._meta_scroll.setWidgetResizable(True)
        self._meta_widget = QtWidgets.QWidget()
        self._meta_scroll.setWidget(self._meta_widget)
        self._inner_splitter.addWidget(self._meta_scroll)
        
        splitter.addWidget(self._inner_splitter)
        
        splitter.setSizes([400, 600])
        self._inner_splitter.setSizes([350, 250])
        layout.addWidget(splitter, 1)
        
        btn_layout = QtWidgets.QHBoxLayout()
        self._status_label = QtWidgets.QLabel()
        btn_layout.addWidget(self._status_label)
        btn_layout.addStretch()
        
        self._merge_single_btn = QtWidgets.QPushButton("Merge Selected Group")
        self._merge_single_btn.clicked.connect(self._on_merge_single)
        self._merge_single_btn.setEnabled(False)
        btn_layout.addWidget(self._merge_single_btn)
        
        self._merge_btn = QtWidgets.QPushButton("Merge Checked Groups")
        self._merge_btn.clicked.connect(self._on_merge)
        btn_layout.addWidget(self._merge_btn)
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
    def _populate(self) -> None:
        self._tree.clear()
        self._primary_selections.clear()
        
        for group in self._duplicate_groups:
            norm_name = group["norm_name"]
            probes = group["probes"]
            
            group_item = QtWidgets.QTreeWidgetItem(self._tree)
            group_item.setText(0, "")
            group_item.setText(2, f"Group: {norm_name} ({len(probes)} items)")
            group_item.setFlags(group_item.flags() | QtCore.Qt.ItemIsUserCheckable)
            group_item.setCheckState(0, QtCore.Qt.Unchecked)
            group_item.setData(0, QtCore.Qt.UserRole, norm_name)
            
            # Button group for the radio buttons in this group
            bg = QtWidgets.QButtonGroup(self)
            
            # Find probe with longest name
            longest_probe = max(probes, key=lambda p: len(p.get("chromophore_name", "")))
            
            for p in probes:
                probe_id = p["probe_id"]
                child = QtWidgets.QTreeWidgetItem(group_item)
                child.setText(1, str(probe_id))
                child.setText(2, p.get("chromophore_name", ""))
                child.setText(3, p.get("category", ""))
                child.setText(4, p.get("source", ""))
                child.setText(5, p.get("verification_status", ""))
                
                rb = QtWidgets.QRadioButton()
                rb.setProperty("probe_id", probe_id)
                rb.setProperty("norm_name", norm_name)
                bg.addButton(rb)
                
                rb.toggled.connect(self._on_radio_toggled)
                if probe_id == longest_probe["probe_id"]:
                    rb.setChecked(True)
                
                self._tree.setItemWidget(child, 0, rb)
            
            group_item.setExpanded(True)
            
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        self._update_merge_btn()
            
    def _on_tree_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if column == 0 and item.data(0, QtCore.Qt.UserRole):
            self._update_merge_btn()

    def _on_radio_toggled(self, checked: bool) -> None:
        if not checked:
            return
        rb = self.sender()
        if not isinstance(rb, QtWidgets.QRadioButton):
            return
        
        probe_id = rb.property("probe_id")
        norm_name = rb.property("norm_name")
        self._primary_selections[norm_name] = probe_id
        self._update_merge_btn()
        
    def _update_merge_btn(self) -> None:
        checked_count = 0
        for i in range(self._tree.topLevelItemCount()):
            if self._tree.topLevelItem(i).checkState(0) == QtCore.Qt.Checked:
                checked_count += 1
                
        if checked_count > 0:
            self._merge_btn.setEnabled(True)
            self._merge_btn.setText(f"Merge {checked_count} Checked Group{'s' if checked_count > 1 else ''}")
        else:
            self._merge_btn.setEnabled(False)
            self._merge_btn.setText("Merge Checked Groups")
            
        current = self._tree.currentItem()
        norm_name = None
        if current:
            norm_name = current.data(0, QtCore.Qt.UserRole)
            if not norm_name:
                parent = current.parent()
                if parent:
                    norm_name = parent.data(0, QtCore.Qt.UserRole)
        
        if norm_name:
            self._merge_single_btn.setEnabled(True)
        else:
            self._merge_single_btn.setEnabled(False)

    def _on_tree_selection(self, current: QtWidgets.QTreeWidgetItem | None, previous: QtWidgets.QTreeWidgetItem | None) -> None:
        if not current:
            return
            
        selected_probe_id = None
        norm_name = current.data(0, QtCore.Qt.UserRole)
        if not norm_name:
            parent = current.parent()
            if parent:
                norm_name = parent.data(0, QtCore.Qt.UserRole)
                try:
                    selected_probe_id = int(current.text(1))
                except (ValueError, TypeError):
                    pass
                
        if not norm_name:
            return
            
        self._update_merge_btn()
            
        group = next((g for g in self._duplicate_groups if g["norm_name"] == norm_name), None)
        if not group:
            return
            
        if norm_name not in self._spectra_cache:
            pids = [p["probe_id"] for p in group["probes"]]
            try:
                res = self._client._call("fluorophores.get_spectra_batch", {"probe_ids": pids})
                self._spectra_cache[norm_name] = res.get("spectra", [])
            except Exception as e:
                import logging
                logging.error(f"Failed to fetch spectra for group {norm_name}: {e}")
                self._spectra_cache[norm_name] = []
                
        model = GroupDetailModel(norm_name, group["probes"], self._spectra_cache[norm_name], selected_probe_id=selected_probe_id)
        from chisurf.gui.autoform import AutoForm
        
        class ViewProxy:
            def __init__(self, model, view_func):
                self._model = model
                self._view_func = view_func
            def view_spec(self):
                return self._view_func()
            def __getattr__(self, name):
                return getattr(self._model, name)
                
        plot_form = AutoForm(ViewProxy(model, model.plot_view_spec))
        meta_form = AutoForm(ViewProxy(model, model.meta_view_spec))
        
        self._plot_scroll.setWidget(plot_form)
        self._meta_scroll.setWidget(meta_form)

    def _on_merge_single(self) -> None:
        current = self._tree.currentItem()
        if not current:
            return
        norm_name = current.data(0, QtCore.Qt.UserRole)
        if not norm_name:
            parent = current.parent()
            if parent:
                norm_name = parent.data(0, QtCore.Qt.UserRole)
        if norm_name:
            self._on_merge(norm_names_to_merge=[norm_name])

    def _on_merge(self, norm_names_to_merge=None) -> None:
        if not self._primary_selections:
            return
            
        if norm_names_to_merge is None:
            norm_names_to_merge = []
            for i in range(self._tree.topLevelItemCount()):
                item = self._tree.topLevelItem(i)
                if item.checkState(0) == QtCore.Qt.Checked:
                    norm_names_to_merge.append(item.data(0, QtCore.Qt.UserRole))
                    
        if not norm_names_to_merge:
            return
            
        success_count = 0
        error_count = 0
        merged_norm_names = []
        
        # Disable UI during merge
        self._merge_btn.setEnabled(False)
        self._merge_single_btn.setEnabled(False)
        self.setCursor(QtCore.Qt.WaitCursor)
        
        try:
            for group in self._duplicate_groups:
                norm_name = group["norm_name"]
                if norm_name not in norm_names_to_merge:
                    continue
                    
                primary_id = self._primary_selections.get(norm_name)
                if not primary_id:
                    continue
                    
                probes = group["probes"]
                duplicate_ids = [p["probe_id"] for p in probes if p["probe_id"] != primary_id]
                
                if duplicate_ids:
                    try:
                        self._client._call(
                            "fluorophores.merge",
                            {
                                "primary_id": primary_id,
                                "duplicate_ids": duplicate_ids
                            }
                        )
                        success_count += 1
                        merged_norm_names.append(norm_name)
                    except Exception as e:
                        import logging
                        logging.error(f"Failed to merge group {norm_name}: {e}")
                        error_count += 1
                        
            if success_count > 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Merge Complete",
                    f"Successfully merged {success_count} group(s)."
                )
                
                for merged_name in merged_norm_names:
                    self._duplicate_groups = [g for g in self._duplicate_groups if g["norm_name"] != merged_name]
                    for i in range(self._tree.topLevelItemCount() - 1, -1, -1):
                        item = self._tree.topLevelItem(i)
                        if item.data(0, QtCore.Qt.UserRole) == merged_name:
                            self._tree.takeTopLevelItem(i)
                            break
                            
            if error_count > 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Merge Errors",
                    f"Failed to merge {error_count} group(s). Check logs for details."
                )
                
            if not self._duplicate_groups:
                self.accept()
            else:
                self._update_merge_btn()
                self._plot_scroll.setWidget(QtWidgets.QWidget())
                self._meta_scroll.setWidget(QtWidgets.QWidget())
            
        finally:
            self.setCursor(QtCore.Qt.ArrowCursor)
