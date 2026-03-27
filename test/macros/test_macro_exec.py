import sys
import pathlib
import importlib
import chisurf
from chisurf.gui.main import Main

# Create a Main instance
main = Main()

# Path to the pong_game plugin's __init__.py file
plugin_path = pathlib.Path(chisurf.plugins.__file__).parent / "pong_game" / "__init__.py"
print(f"Testing macro execution with file: {plugin_path}")

try:
    # Run the macro using the 'exec' executor
    main.onRunMacro(filename=plugin_path, executor='exec')
    print("Macro executed successfully!")
except Exception as e:
    print(f"Error executing macro: {e}")
    import traceback
    traceback.print_exc()