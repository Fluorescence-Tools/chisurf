"""
Browser Plugin

This plugin provides a web browser interface within the ChiSurf application.
It allows users to:
1. Navigate to URLs
2. Reload pages
3. Return to a home page
4. View web content within the application

The browser has security features like blocking non-file URLs to ensure safe browsing.
It's useful for viewing documentation, tutorials, or other web-based resources
without leaving the ChiSurf environment.
"""

from .wizard import *

name = "Tools:Miscellaneous:Browser"
menu_hidden = True
deprecated = True
deprecation_message = (
    "The embedded Browser plugin is deprecated. "
    "Notebook/help links now open in the system browser."
)
