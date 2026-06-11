"""Single Molecule Burst Selection Wizard."""

from . import BurstSelectionTool

if __name__ == "plugin":
    # You can customize the visibility of UI elements here
    brick_mic_wiz = BurstSelectionTool(
        show_channel_selection=True,
        show_clear_button=False,
        show_decay_button=False,
        show_filter_button=False
    )
    brick_mic_wiz.show()


if __name__ == '__main__':
    import sys

    from qtpy import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    # You can customize the visibility of UI elements here
    brick_mic_wiz = BurstSelectionTool(
        show_channel_selection=True,
        show_clear_button=False,
        show_decay_button=False,
        show_filter_button=False
    )
    brick_mic_wiz.setWindowTitle('BRICK-Mic')
    brick_mic_wiz.show()
    sys.exit(app.exec_())
