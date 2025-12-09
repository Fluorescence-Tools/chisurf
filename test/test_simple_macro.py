import sys
import os

# Create a simple macro file
simple_macro = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple_macro.py")
with open(simple_macro, 'w') as f:
    f.write("""
# A simple macro that doesn't use relative imports
print("Simple macro executed successfully!")
x = 10
y = 20
result = x + y
print(f"Result: {result}")
""")

print(f"Created simple macro: {simple_macro}")

# Create a globals dictionary similar to what onRunMacro would create
globals_dict = {
    "__name__": "__main__",
    "__file__": simple_macro
}

# Get the directory of the file
macro_dir = os.path.dirname(simple_macro)

# Temporarily add the macro directory to sys.path for relative imports
original_sys_path = sys.path.copy()
if macro_dir not in sys.path:
    sys.path.insert(0, macro_dir)

try:
    # Determine if this is part of a package (it's not, but we'll run the same code)
    if '\\plugins\\' in simple_macro:
        # Extract package name from path
        parts = simple_macro.split('\\plugins\\')
        if len(parts) > 1:
            plugin_path = parts[1].split('\\')
            if len(plugin_path) > 0:
                package_name = plugin_path[0]
                # Set __package__ for relative imports to work
                globals_dict["__package__"] = f"chisurf.plugins.{package_name}"
                print(f"Set __package__ to: {globals_dict['__package__']}")

    # Execute the file
    with open(simple_macro, 'rb') as file:
        code = compile(file.read(), simple_macro, 'exec')
        print("Compiled code successfully")
        exec(code, globals_dict)
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore the original sys.path
    sys.path = original_sys_path
    print("Restored original sys.path")
    
    # Clean up the temporary file
    if os.path.exists(simple_macro):
        os.remove(simple_macro)
        print(f"Removed temporary file: {simple_macro}")