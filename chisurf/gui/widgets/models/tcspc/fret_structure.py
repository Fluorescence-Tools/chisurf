from __future__ import annotations

from typing import TYPE_CHECKING
import chisurf as cs
from qtpy import QtWidgets, QtCore, QtGui

from chisurf.core.structure import Structure
from chisurf.core.models.tcspc.fret_structure import FRETStructure
from chisurf.gui.widgets.models.tcspc.lifetime import LifetimeWidget, LifetimeModelWidgetBase
from chisurf.gui.widgets.models.tcspc.anisotropy import AnisotropyWidget
from chisurf.gui.widgets.models.tcspc import plot_cls_dist_default
from chisurf.gui.widgets.models.tcspc import kappa2_helpers
from chisurf.gui.widgets.fluorescence.av import AVProperties


if TYPE_CHECKING:
    from chisurf.core.fitting.fit import Fit


class FRETStructureWidget(FRETStructure, LifetimeModelWidgetBase):
    """GUI widget for FRET Structure fit model."""

    plot_classes = plot_cls_dist_default

    def get_parameter_widgets(self):
        """Get all parameter widgets for this model.

        Returns
        -------
        list
            List of parameter widgets.
        """
        widgets = super().get_parameter_widgets() if hasattr(super(), 'get_parameter_widgets') else []
        if hasattr(self, '_orientation_widget'):
            widgets.append(self._orientation_widget)
        widgets.append(self._fret_parameters_widget)
        return widgets

    def finalize(self):
        """Finalize the component state."""
        super().finalize()
        fractions = self.amplitudes
        s = ""
        for f in fractions:
            s += "%.3f\n" % f
        self.fraction_widget.setPlainText(s)
        self.donor.update()

    def clear(self):
        """Clear all loaded structures and fractions."""
        FRETStructure.clear(self)
        self.pdb_line_widget.setPlainText("")
        self.fraction_widget.setPlainText("")

    def onOpenFilenames(self):
        """Open file dialog to load PDB structures."""
        filenames = cs.gui.widgets.open_files("Open PDBs", '*.pdb')
        st = ""
        for filename in filenames:
            s = Structure(filename)
            self.append(s)
            st += filename + "\n"
        self.pdb_line_widget.appendPlainText(st)
        self.folder_line.setText(str(cs.working_path))

    @property
    def linker_width_1(self) -> float:
        """Linker width of donor."""
        return float(self.donor_av.linker_width)

    @linker_width_1.setter
    def linker_width_1(self, v: float):
        """Linker width of donor."""
        self.donor_av.linker_width = float(v)

    @property
    def linker_width_2(self) -> float:
        """Linker width of acceptor."""
        return float(self.acceptor_av.linker_width)

    @linker_width_2.setter
    def linker_width_2(self, v: float):
        """Linker width of acceptor."""
        self.acceptor_av.linker_width = float(v)

    @property
    def radius1_1(self) -> float:
        """Radius of donor."""
        return float(self.donor_av.radius_1)

    @radius1_1.setter
    def radius1_1(self, v: float):
        """Radius of donor."""
        self.donor_av.radius_1 = float(v)

    @property
    def radius1_2(self) -> float:
        """Radius of acceptor."""
        return float(self.acceptor_av.radius_1)

    @radius1_2.setter
    def radius1_2(self, v: float):
        """Radius of acceptor."""
        self.acceptor_av.radius_1 = float(v)

    @property
    def linker_length_1(self) -> float:
        """Linker length of donor."""
        return float(self.donor_av.linker_length)

    @linker_length_1.setter
    def linker_length_1(self, v: float):
        """Linker length of donor."""
        self.donor_av.linker_length = float(v)

    @property
    def linker_length_2(self) -> float:
        """Linker length of acceptor."""
        return float(self.acceptor_av.linker_length)

    @linker_length_2.setter
    def linker_length_2(self, v: float):
        """Linker length of acceptor."""
        self.acceptor_av.linker_length = float(v)

    @property
    def atom_name_1(self) -> str:
        """Atom name of donor labeling residue."""
        return str(self.donor_atom.text())

    @atom_name_1.setter
    def atom_name_1(self, v: str):
        """Atom name of donor labeling residue."""
        self.donor_atom.setText(str(v))

    @property
    def atom_name_2(self) -> str:
        """Atom name of acceptor labeling residue."""
        return str(self.acceptor_atom.text())

    @atom_name_2.setter
    def atom_name_2(self, v: str):
        """Atom name of acceptor labeling residue."""
        self.acceptor_atom.setText(str(v))

    @property
    def res_1(self) -> int:
        """Residue index of donor labeling residue."""
        return int(self.donor_res.value())

    @res_1.setter
    def res_1(self, v: int):
        """Residue index of donor labeling residue."""
        self.donor_res.setValue(int(v))

    @property
    def res_2(self) -> int:
        """Residue index of acceptor labeling residue."""
        return int(self.acceptor_res.value())

    @res_2.setter
    def res_2(self, v: int):
        """Residue index of acceptor labeling residue."""
        self.acceptor_res.setValue(int(v))

    def __init__(self, fit: Fit, **kwargs):
        """Initialize the FRETStructureWidget."""
        # Instantiate widgets that properties map to before parent initializations
        self.donor_atom = QtWidgets.QLineEdit()
        self.donor_res = QtWidgets.QSpinBox()
        self.donor_res.setMaximum(1000)
        self.donor_av = AVProperties()

        self.acceptor_atom = QtWidgets.QLineEdit()
        self.acceptor_res = QtWidgets.QSpinBox()
        self.acceptor_res.setMaximum(1000)
        self.acceptor_av = AVProperties()

        self.folder_line = QtWidgets.QLineEdit()
        self.folder_line.setEnabled(False)
        self.pdb_line_widget = QtWidgets.QPlainTextEdit()
        self.fraction_widget = QtWidgets.QPlainTextEdit()

        self.donor = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)',
            short='D',
            name='donors'
        )

        anisotropy = AnisotropyWidget(
            name='anisotropy',
            short='rL',
            fit=fit,
            model=self,
            **kwargs
        )
        kwargs['anisotropy'] = anisotropy

        # Initialize core FRETStructure model
        FRETStructure.__init__(self, fit=fit, lifetimes=self.donor, **kwargs)

        # Initialize LifetimeModelWidgetBase
        LifetimeModelWidgetBase.__init__(
            self,
            fit=fit,
            **kwargs
        )

        self.layout.addWidget(self.donor)

        # Create parameter widgets
        self._fret_parameters_widget = cs.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters
        )
        self.layout.addWidget(self._fret_parameters_widget)

        # Shared κ² mode controls
        kappa2_helpers.setup_kappa2_controls(self, self.layout)

        # Labels parameters layout
        self.groupBox = QtWidgets.QGroupBox()
        self.groupBox.setTitle("Labels / AV")
        vbox = QtWidgets.QVBoxLayout(self.groupBox)

        # Donor Box
        donor_box = QtWidgets.QGroupBox()
        donor_box.setTitle("Donor")
        donor_lay = QtWidgets.QVBoxLayout(donor_box)
        
        donor_sel_lay = QtWidgets.QHBoxLayout()
        donor_sel_lay.addWidget(QtWidgets.QLabel("Residue:"))
        donor_sel_lay.addWidget(self.donor_res)
        donor_sel_lay.addWidget(QtWidgets.QLabel("Atom:"))
        donor_sel_lay.addWidget(self.donor_atom)
        donor_lay.addLayout(donor_sel_lay)
        donor_lay.addWidget(self.donor_av)
        vbox.addWidget(donor_box)

        # Acceptor Box
        acceptor_box = QtWidgets.QGroupBox()
        acceptor_box.setTitle("Acceptor")
        acceptor_lay = QtWidgets.QVBoxLayout(acceptor_box)
        
        acceptor_sel_lay = QtWidgets.QHBoxLayout()
        acceptor_sel_lay.addWidget(QtWidgets.QLabel("Residue:"))
        acceptor_sel_lay.addWidget(self.acceptor_res)
        acceptor_sel_lay.addWidget(QtWidgets.QLabel("Atom:"))
        acceptor_sel_lay.addWidget(self.acceptor_atom)
        acceptor_lay.addLayout(acceptor_sel_lay)
        acceptor_lay.addWidget(self.acceptor_av)
        vbox.addWidget(acceptor_box)

        self.layout.addWidget(self.groupBox)

        self.anisotropy = anisotropy
        self.layout.addWidget(self.anisotropy)

        # PDB selection UI
        gb = QtWidgets.QGroupBox()
        gb.setTitle('PDBs')
        l = QtWidgets.QVBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)

        l1 = QtWidgets.QHBoxLayout()
        l1.setSpacing(0)
        l1.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel('Folder:')
        open_folder = QtWidgets.QPushButton()
        open_folder.setText('...')
        open_folder.clicked.connect(self.onOpenFilenames)
        l1.addWidget(label)
        l1.addWidget(self.folder_line)
        l1.addWidget(open_folder)
        l.addLayout(l1)

        l.addWidget(self.pdb_line_widget)
        l.addWidget(QtWidgets.QLabel('Fractions:'))
        l.addWidget(self.fraction_widget)
        gb.setLayout(l)
        
        self.layout.addWidget(gb)

        # Synchronize UI values with defaults from model initialization
        self.res_1 = self.res_1
        self.res_2 = self.res_2
        self.atom_name_1 = self.atom_name_1
        self.atom_name_2 = self.atom_name_2
        self.linker_length_1 = self.linker_length_1
        self.linker_length_2 = self.linker_length_2
        self.linker_width_1 = self.linker_width_1
        self.linker_width_2 = self.linker_width_2
        self.radius1_1 = self.radius1_1
        self.radius1_2 = self.radius1_2

    def set_state(self, state: dict) -> None:
        """Restore state from snapshot and update GUI elements."""
        FRETStructure.set_state(self, state)
        # Refresh the PDB text box
        self.pdb_line_widget.setPlainText("\n".join(self.filenames))
        # Refresh the fractions text box
        self.finalize()
        # Synchronize UI values with defaults from model
        self.res_1 = self.res_1
        self.res_2 = self.res_2
        self.atom_name_1 = self.atom_name_1
        self.atom_name_2 = self.atom_name_2
        self.linker_length_1 = self.linker_length_1
        self.linker_length_2 = self.linker_length_2
        self.linker_width_1 = self.linker_width_1
        self.linker_width_2 = self.linker_width_2
        self.radius1_1 = self.radius1_1
        self.radius1_2 = self.radius1_2

    def _install_code_badge(self):
        """Install a code badge for dev mode source jumping."""
        try:
            import chisurf.core.settings
            if not cs.core.settings.is_dev_mode():
                return
            if hasattr(self, '_chisurf_code_badge_installed'):
                return
            from chisurf.gui.devtools.source_jump import resolve_object_source
            from chisurf.gui.widgets.code_badge import install_code_badge
            resolver = lambda: resolve_object_source(self)
            install_code_badge(self, resolver, corner='top-right', margin=4)
            self._chisurf_code_badge_installed = True
        except Exception:
            pass
