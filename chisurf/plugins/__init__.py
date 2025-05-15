import os
import sys
import pathlib

# Define the user plugins directory
user_plugins_dir = pathlib.Path.home() / '.chisurf' / 'plugins'

# Create the directory if it doesn't exist
user_plugins_dir.mkdir(parents=True, exist_ok=True)

# Add the user plugins directory to the module's __path__
if str(user_plugins_dir) not in __path__:
    __path__.append(str(user_plugins_dir))
