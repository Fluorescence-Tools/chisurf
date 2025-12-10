# TTTR Image Browser Plugin

This plugin provides an interactive browser for TTTR (time-tagged time-resolved) data.

## Features

- Browse TTTR files in a folder and show intensity images for all DetectorWizard-defined windows.
- Star-rating (0–3) and annotation per file.
- Filtering and sorting by rating.
- Export of selected files and optional DOCX reports compatible with the Trace Browser.

## Usage

- Open the plugin from the ChiSurf **Plugins → TTTR:Image Browser** menu.
- Use the DetectorWizard page to define detectors and time windows.
- Choose a folder with TTTR files and inspect the generated images.
- Use the rating, annotation, and export controls to curate datasets.

The source code lives in `chisurf/plugins/tttr_image_browser` and shares components with the Trace Browser plugin.
