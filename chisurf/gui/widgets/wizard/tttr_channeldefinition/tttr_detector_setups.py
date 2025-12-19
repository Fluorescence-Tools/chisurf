import json
import pathlib

from qtpy.QtWidgets import QMessageBox, QCheckBox

from chisurf.settings.path_utils import get_path
from chisurf.settings.file_utils import safe_open_file


DETECTOR_SETUPS_FILE = get_path('settings') / 'detector_setups.json'


def load_detector_setups(file_path=None):
    """Load detector setups from the central settings file or a custom file.

    If the central setups file does not exist, inform the user how to create it
    and allow suppressing this warning in the future.
    """
    path = pathlib.Path(file_path or DETECTOR_SETUPS_FILE)

    try:
        import chisurf.settings
        show_warning = bool(chisurf.settings.cs_settings.get('warn_missing_detector_setups', True))
    except Exception:
        show_warning = True

    if not path.exists():
        is_default = (file_path is None) or (path == DETECTOR_SETUPS_FILE)
        app_running = False
        try:
            from qtpy.QtWidgets import QApplication
            app_running = QApplication.instance() is not None
        except Exception:
            app_running = False

        try:
            import chisurf as _chisurf_mod
            if is_default and getattr(_chisurf_mod, "__startup_in_progress__", False):
                try:
                    _chisurf_mod.__pending_startup_onboarding__ = True
                except Exception:
                    pass
                return {"setups": {}}
        except Exception:
            pass

        if show_warning and is_default and app_running:
            msg = QMessageBox()
            msg.setWindowTitle("Detector setups file not found")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Detector setups file was not found:\n{str(path)}")
            msg.setInformativeText(
                "You can create it by saving a setup from the Detector Wizard.\n"
                "Use the 'Save Settings' button to store your configuration.\n"
                "Alternatively, choose an existing JSON with the '...' button."
            )
            try:
                cb = QCheckBox("Don't show this warning again")
                msg.setCheckBox(cb)
            except Exception:
                cb = None

            open_btn = msg.addButton("Open Detector Wizard", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Ok)
            msg.exec_()

            try:
                if cb is not None and cb.isChecked():
                    from chisurf.settings.settings_utils import set_warn_missing_detector_setups as _set_warn
                    _set_warn(False)
                    try:
                        import chisurf.settings
                        chisurf.settings.cs_settings['warn_missing_detector_setups'] = False
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                if msg.clickedButton() is open_btn:
                    from .tttr_channel_definition import DetectorWizard
                    wiz = DetectorWizard()
                    wiz.exec_()
                    if path.exists():
                        return safe_open_file(
                            file_path=path,
                            processor=json.load,
                            default_value={"setups": {}}
                        )
            except Exception:
                pass

        return {"setups": {}}

    return safe_open_file(
        file_path=path,
        processor=json.load,
        default_value={"setups": {}}
    )


def save_detector_setups(setups_data, file_path=None, replace=False):
    """Save detector setups to the central settings file or a custom file.

    Args:
        setups_data: The data to save
        file_path: Optional custom path to save to. If None, uses DETECTOR_SETUPS_FILE.
    """
    try:
        save_path = file_path or DETECTOR_SETUPS_FILE

        try:
            if replace:
                updated_data = setups_data
            elif pathlib.Path(save_path).exists() and pathlib.Path(save_path).stat().st_size > 0:
                existing_data = load_detector_setups(save_path)
                updated_data = existing_data if isinstance(existing_data, dict) else {}
                if isinstance(setups_data, dict):
                    for k, v in setups_data.items():
                        if k == "setups":
                            updated_data.setdefault("setups", {})
                            if isinstance(v, dict):
                                updated_data["setups"].update(v)
                        else:
                            updated_data[k] = v
                else:
                    updated_data = setups_data
            else:
                updated_data = setups_data
        except Exception:
            updated_data = setups_data

        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(updated_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving detector setups: {e}")
        return False
