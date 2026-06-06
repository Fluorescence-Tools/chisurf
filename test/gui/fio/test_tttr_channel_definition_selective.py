from pathlib import Path
import tempfile

from qtpy.QtWidgets import QWizard, QLabel, QVBoxLayout, QWidget
from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage


def test_detector_wizard_page_instantiation(qapp):
    wizard = QWizard()
    page = DetectorWizardPage()
    wizard.addPage(page)
    assert page is not None


def test_instruction_widget_creation(qapp):
    instructions = QWidget()
    layout = QVBoxLayout(instructions)
    label = QLabel(
        "Test the selective reading behavior:\n"
        "1. Click 'Read TTTR' button\n"
        "2. Select a file based on its type:\n"
        "   - .set files: Should only update microtime\n"
        "   - .spc files: Should only update macrotime\n"
        "   - Other files: Should update both"
    )
    layout.addWidget(label)
    assert instructions is not None
    assert label.text() != ""


def test_set_file_creation():
    with tempfile.TemporaryDirectory() as temp_dir:
        set_file = Path(temp_dir) / "test.set"
        content = """
        [SP_SYN_FQ,F,-50.98]
        [SP_TAC_TC,F,1.83e-11]
        [SP_TAC_R,F,5.0e-8]
        [SP_ADC_RE,I,4096]
        """
        set_file.write_text(content)
        assert set_file.exists()
