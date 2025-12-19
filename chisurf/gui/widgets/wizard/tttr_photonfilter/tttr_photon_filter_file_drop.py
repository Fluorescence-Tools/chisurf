import pathlib

import tttrlib

import chisurf.gui.decorators
from chisurf.gui import QtWidgets

from .tttr_photon_filter_support import ProgressWindow


def install_file_drop(page):
    def after_file_drop():
        """
        Callback to load *all* TTTR files after they're dropped or specified.
        Ensures that files with restricted extensions require explicit file type selection.
        If a folder is dropped, all files with supported extensions (found directly in that folder)
        are added.
        Uses tttrlib.inferTTTRFileType to automatically detect file types when possible.
        """
        # Generate allowed extensions dynamically from tttrlib
        allowed_extensions = {
            f".{ext.lower()}" if not ext.startswith('.') else ext.lower()
            for ext in tttrlib.TTTR.get_supported_container_names()
        }

        # Expand directories: if an entry in tttr_filenames is a folder,
        # replace it with all files (in that folder only) with allowed extensions.
        expanded_files = []
        for path_str in page.settings['tttr_filenames']:
            p = pathlib.Path(path_str).resolve()
            if p.is_dir():
                for child in p.iterdir():
                    if child.is_file() and child.suffix.lower() in allowed_extensions:
                        expanded_files.append(str(child.resolve()))
            else:
                expanded_files.append(str(p))
        # IMPORTANT: mutate the existing list in-place to preserve the drag/drop injector reference
        lst = page.settings.get('tttr_filenames')
        if isinstance(lst, list):
            lst[:] = expanded_files

        # List of restricted extensions requiring manual selection (if needed)
        RESTRICTED_EXTENSIONS = [".spc"]  # Extend or modify as required

        requires_filetype_selection = False
        restricted_files = []

        for fn in page.settings['tttr_filenames']:
            p = pathlib.Path(fn).resolve()
            file_extension = p.suffix.lower()

            # Check if the file extension requires explicit selection
            if file_extension in RESTRICTED_EXTENSIONS:
                if page.filetype == "Auto":  # Only warn if no file type is preselected
                    requires_filetype_selection = True
                    restricted_files.append(p.name)

        if requires_filetype_selection:
            QtWidgets.QMessageBox.warning(
                page, "File Type Required",
                "The following files require an explicit file type selection before loading:\n\n"
                + "\n".join(restricted_files)
                + "\n\nPlease select the correct file type from the dropdown menu."
            )
            page.onClearFiles()
            return  # Prevent loading any files

        # Get file type from the selected setup
        file_type = page.filetype
        # If no setup is selected or the setup doesn't have a file type,
        # we've already shown a warning in the filetype property

        # Proceed with loading the files and showing progress
        total_files = len(page.settings['tttr_filenames'])
        progress_window = ProgressWindow(title="Loading Files", message="Processing files...",
                                         max_value=total_files, parent=page)
        progress_window.show()

        for i, fn in enumerate(page.settings['tttr_filenames'], start=1):
            p = pathlib.Path(fn).resolve()
            p_str = str(p)

            if p_str not in page.tttr_objects:
                if p.exists() and p.is_file():
                    file_type = page.filetype
                    try:
                        if isinstance(file_type, str):
                            page.tttr_objects[p_str] = tttrlib.TTTR(p_str, file_type)
                        elif p.suffix.lower() not in RESTRICTED_EXTENSIONS:
                            # Use inferTTTRFileType for better auto-detection
                            file_type_int = tttrlib.inferTTTRFileType(p_str)
                            if file_type_int is not None and file_type_int >= 0:
                                page.tttr_objects[p_str] = tttrlib.TTTR(p_str, file_type_int)
                            else:
                                # Fall back to default auto-detection if inference fails
                                page.tttr_objects[p_str] = tttrlib.TTTR(p_str)
                    except Exception as e:
                        progress_window.close()
                        QtWidgets.QMessageBox.critical(
                            page,
                            "Error Loading File",
                            f"Failed to load file '{p.name}' with the selected setup.\n\n"
                            f"Error: {str(e)}\n\n"
                            f"Please check that you have selected the correct setup for this file type."
                        )
                        page.onClearFiles()
                        return  # Exit early to prevent undefined state

            progress_window.set_value(i)

        progress_window.close()

        n_files = len(page.settings['tttr_filenames'])
        page.spinBox_4.setMaximum(n_files - 1)
        if n_files > 0:
            page.spinBox_4.setValue(n_files - 1)
        page.read_tttr()

    # Inject file-drop logic
    page.textEdit.setVisible(False)
    # Expose the drop handler so external code (e.g., batch processing) can reuse the standard flow
    page._after_file_drop = after_file_drop
    chisurf.gui.decorators.lineEdit_dragFile_injector(
        page.lineEdit, call=after_file_drop, target=page.settings['tttr_filenames']
    )
