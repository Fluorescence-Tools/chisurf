# PTU Header Editor Plugin

This plugin provides a comprehensive tool for viewing and editing header tags in PicoQuant Time-Tagged Time-Resolved 
(PTU) files.

## Features

- Open and inspect PTU files
- View all header tags and their values in a tabular format
- Edit existing tag values
- Add new custom tags to the header
- Remove unwanted tags
- Toggle between table and JSON views
- Save modified PTU files with updated header information
- Preserve all event data while modifying header information

## Overview

The PTU Header Editor plugin enables users to view and modify the metadata contained in PicoQuant Time-Tagged 
Time-Resolved (PTU) files. PTU files contain important experimental metadata in their headers, which may  
need correction or enhancement for proper analysis.

This plugin provides an intuitive table-based interface that displays tag names, types, values, and indices. Users can 
modify any field and see the changes in real-time. A JSON view is also available to see the complete header structure. 
The plugin preserves all event data from the original file while allowing complete customization of the header 
information.

The editor supports all standard PTU tag types and provides appropriate validation when editing values to ensure the 
integrity of the file structure.

## Requirements

- Python packages:
  - PyQt5
  - tttrlib
  - json
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: TTTR > PTU Header editor
2. Open a PTU file:
   - Click the "Open PTU File" button
   - Select a PTU file from the file dialog
3. View and edit header tags:
   - The table displays all tags with their names, types, values, and indices
   - Double-click on a value cell to edit it
   - Changes are validated based on the tag type
4. Add new tags:
   - Click the "Add Tag" button
   - Enter the tag name, type, and value in the dialog
5. Remove tags:
   - Select a tag row in the table
   - Click the "Remove Tag" button
6. Toggle view mode:
   - Click the "Toggle JSON View" button to switch between table and JSON views
7. Save changes:
   - Click the "Save Modified PTU" button
   - Choose a location to save the modified PTU file

## Applications

The PTU Header Editor plugin can be used for:
- Correcting metadata in experimental PTU files
- Adding missing information to PTU headers
- Preparing files for specialized analysis
- Troubleshooting issues with file metadata
- Educational purposes to understand PTU file structure
- Standardizing metadata across multiple files
- Adding custom tags for specific analysis workflows

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.