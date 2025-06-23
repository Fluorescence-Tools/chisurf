# Browser Plugin

This plugin provides a web browser interface within the ChiSurf application.

## Features

- Navigate to URLs
- Reload pages
- Return to a home page
- View web content within the application
- Security features like blocking non-file URLs to ensure safe browsing

## Overview

The Browser plugin allows users to view web content without leaving the ChiSurf environment. It's particularly useful 
for accessing documentation, tutorials, or other web-based resources related to data analysis and visualization.

The integrated browser provides a convenient way to reference online materials while working with ChiSurf, enhancing 
productivity by eliminating the need to switch between applications.

## Requirements

- Python packages:
  - PyQt5
  - QtWebEngineWidgets

## Usage

1. Launch the plugin from the ChiSurf menu: Miscellaneous > Browser
2. Enter a URL in the address bar
3. Use the navigation buttons to:
   - Go back to previous pages
   - Go forward to next pages
   - Reload the current page
   - Return to the home page
4. View web content in the browser window

## Security Features

The browser implements security measures to ensure safe browsing:
- Blocking of non-file URLs by default
- Restricted JavaScript execution
- Limited access to local resources

These features help protect users from potentially harmful web content while still providing access to necessary 
resources.

## Applications

- Viewing online documentation for ChiSurf and related tools
- Accessing tutorials and educational resources
- Referencing scientific literature and databases
- Viewing local HTML documentation files
- Accessing web-based analysis tools

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.