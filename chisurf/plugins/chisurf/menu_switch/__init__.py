"""
Menu Switch

This plugin provides a simple toggle to switch between the traditional menu bar 
and the modern ribbon interface in ChiSurf.

Features:
- One-click switching between menu and ribbon interfaces
- Automatic state detection and switching
- Persistent preference saving
- Seamless transition without requiring restart

The Menu Switch plugin allows users to easily toggle between the traditional
menu bar interface and the modern ribbon interface based on their preference
or workflow requirements. The current interface state is automatically saved
and restored on application startup.

This plugin is particularly useful for users who want to quickly switch between
interfaces for different tasks or for those who are evaluating which interface
works best for their workflow.
"""

name = "Setup:Menu Switch"

# Import the main functionality
from .menu_switch import run

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Execute the menu switch functionality
    run()
