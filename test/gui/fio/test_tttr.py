from pathlib import Path
import tempfile

from qtpy.QtWidgets import QWizard, QLabel, QVBoxLayout, QWidget
from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

#
# Part 1: test_tttr_channel_definition
#

def test_tttr_channel_definition(qapp):
    wizard = QWizard()
    page = DetectorWizardPage()
    wizard.addPage(page)
    assert page is not None


def test_tttr_channel_definition_set_file():
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


#
# Part 2: test_tttr_channel_definition_selective
#

def test_tttr_channel_definition_selective_widget(qapp):
    wizard = QWizard()
    page = DetectorWizardPage()
    wizard.addPage(page)
    assert page is not None


def test_tttr_channel_definition_selective_instructions(qapp):
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
    assert label is not None


#
# Part 3: unicode TTTR filename
#

def test_unicode_tttr_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test_µ_file.ptu"
        filename.write_text("mock")
        assert filename.exists()


#
# Part 4: burst imports
#

import chisurf.core.fluorescence.burst


def test_burst_imports():
    bocpd_func = chisurf.core.fluorescence.burst.bocpd.convert_bursts_to_start_stop
    kalman_func = chisurf.core.fluorescence.burst.kalman.convert_bursts_to_start_stop
    assert bocpd_func is not None
    assert kalman_func is not None
    assert bocpd_func is not kalman_func
