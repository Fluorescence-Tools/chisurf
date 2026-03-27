#!/usr/bin/env python
"""
Simple test script to verify that the light.qss style is being applied.
This script will print the current style sheet setting from the configuration.
"""

import os
import sys
import pathlib

# Add the chisurf directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import chisurf.settings

def main():
    # Print the current style sheet setting
    print(f"Current style sheet setting: {chisurf.settings.cs_settings['gui']['style_sheet']}")
    
    # Get the path to the style sheet file
    style_file = pathlib.Path(chisurf.__file__).parent / "gui/styles" / chisurf.settings.cs_settings['gui']['style_sheet']
    print(f"Style sheet file path: {style_file}")
    
    # Check if the file exists
    if style_file.exists():
        print(f"Style sheet file exists: Yes")
        # Print the content of the style sheet file
        with open(style_file, 'r') as f:
            content = f.read()
        print(f"Style sheet content (first 100 chars): {content[:100]}")
    else:
        print(f"Style sheet file exists: No")

if __name__ == "__main__":
    main()