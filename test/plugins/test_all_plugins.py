import sys
import importlib
import pkgutil
import chisurf.plugins

# Get a list of all available plugins
print("Available plugins:")
plugin_count = 0
for _, name, _ in pkgutil.iter_modules(chisurf.plugins.__path__):
    plugin_count += 1
    print(f"  - {name}")

print(f"\nFound {plugin_count} plugins. Testing each one...\n")

# Try to import each plugin
success_count = 0
failure_count = 0
for _, name, _ in pkgutil.iter_modules(chisurf.plugins.__path__):
    module_path = f"chisurf.plugins.{name}"
    print(f"Testing plugin: {name}")
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        print(f"  ✓ Successfully imported {module_path}")
        
        # Don't actually initialize the plugin by setting __name__ to "plugin"
        # as that would open UI windows for each plugin
        
        success_count += 1
    except Exception as e:
        print(f"  ✗ Error loading plugin {module_path}: {e}")
        failure_count += 1

print(f"\nTest complete: {success_count} plugins loaded successfully, {failure_count} failed")