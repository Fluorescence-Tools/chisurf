import sys
import pathlib
import os

# Path to the pong_game plugin's __init__.py file
plugin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chisurf", "plugins", "pong_game")
init_file = os.path.join(plugin_dir, "__init__.py")
print(f"Testing relative import with file: {init_file}")

# Create a globals dictionary similar to what onRunMacro would create
globals_dict = {
    "__name__": "__main__",
    "__file__": init_file
}

# Get the directory of the file
macro_dir = os.path.dirname(init_file)

# Temporarily add the macro directory to sys.path for relative imports
original_sys_path = sys.path.copy()
if macro_dir not in sys.path:
    sys.path.insert(0, macro_dir)

try:
    # Determine if this is part of a package
    if '\\plugins\\' in init_file:
        # Extract package name from path
        parts = init_file.split('\\plugins\\')
        if len(parts) > 1:
            plugin_path = parts[1].split('\\')
            if len(plugin_path) > 0:
                package_name = plugin_path[0]
                # Set __package__ for relative imports to work
                globals_dict["__package__"] = f"chisurf.plugins.{package_name}"
                print(f"Set __package__ to: {globals_dict['__package__']}")

    # Execute the file
    with open(init_file, 'rb') as file:
        code = compile(file.read(), init_file, 'exec')
        print("Compiled code successfully")
        
        # We'll just check if the import would work, not actually execute the full code
        # which would try to create GUI elements
        try:
            # Create a modified version of the code that just imports the module
            import_test_code = """
try:
    from .pong_game import PongGameWidget
    print("Relative import successful!")
except ImportError as e:
    print(f"Relative import failed: {e}")
"""
            exec(compile(import_test_code, "<string>", 'exec'), globals_dict)
        except Exception as e:
            print(f"Error during import test: {e}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore the original sys.path
    sys.path = original_sys_path
    print("Restored original sys.path")