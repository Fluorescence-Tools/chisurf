# ChiSurf Plugin Manual Generator

This script generates a comprehensive manual for all ChiSurf plugins by extracting and formatting their docstrings.

## Features

- Automatically discovers all installed plugins in the ChiSurf application
- Extracts docstrings and plugin names from each plugin
- Organizes plugins by category based on their menu structure
- Creates a well-formatted Markdown document with:
  - Title and introduction
  - Table of contents with links to each plugin section
  - Detailed documentation for each plugin
  - Proper formatting of paragraphs and bullet points
- Saves the manual to the `manual` directory in the ChiSurf repository

## Usage

To generate the plugin manual, simply run:

```bash
python manual\scripts\create_plugin_manual.py
```

The script will:
1. Find all available plugins
2. Extract their docstrings
3. Create a Markdown file with the formatted documentation
4. Save the Markdown file to `manual\plugin_manual.md`
5. Generate a Word document version and save it to `manual\plugin_manual.docx`

## Output

The script generates two output files:

1. **Markdown file** (`manual\plugin_manual.md`): A well-formatted Markdown document with:
   - A title and brief introduction
   - A table of contents organized by plugin category
   - Sections for each plugin with:
     - The plugin name as a heading
     - The module name
     - The full docstring with proper formatting
     - A separator between plugins

2. **Word document** (`manual\plugin_manual.docx`): A formatted Word document that contains the same content as the Markdown file, with:
   - Proper heading styles (H1, H2, H3)
   - Formatted bullet points
   - Italic text for module names
   - Horizontal separators between plugins

The Word document is automatically generated from the Markdown content, so you don't need to use external conversion tools.

## Requirements

The script requires:
- Python 3.6+
- Access to the ChiSurf codebase
- python-docx package (for Word document generation)

The script will still work without python-docx, but it will only generate the Markdown file and not the Word document. If python-docx is not installed, the script will print instructions on how to install it.

To install python-docx:
```bash
pip install python-docx
```

## Customization

If you need to customize the output format or content, you can modify the `create_plugin_manual` function in the script. The main components you might want to adjust include:

- The title and introduction text
- The formatting of plugin sections
- The organization of the table of contents
- The output file location
