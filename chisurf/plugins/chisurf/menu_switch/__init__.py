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
from .menu_switch import run, MenuSwitchWidget

# Note: The plugin no longer auto-executes when loaded.
# Users should explicitly call the menu switch functionality through the GUI
# or by importing and calling run() manually.
