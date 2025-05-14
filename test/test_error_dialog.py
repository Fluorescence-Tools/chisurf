import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Backup the original sys.exit function
original_exit = sys.exit

# Override sys.exit to print a message and not actually exit
def mock_exit(code=0):
    print(f"\nTest completed. Exit code: {code}")
    if code != 0:
        print("Error dialog should have been displayed.")
    return code

# Replace sys.exit with our mock function
sys.exit = mock_exit

# Create a module that will raise an exception when imported
import types
mock_module = types.ModuleType('mock_module')
mock_module.__file__ = 'mock_module.py'

def raise_exception():
    print("Raising test exception...")
    raise Exception("This is a test exception to verify the error dialog")

# Create a mock get_app function that raises an exception
mock_module.get_app = raise_exception

# Add the mock module to sys.modules
sys.modules['chisurf.gui'] = mock_module

# Now import and run the main function
print("Importing main function...")
from chisurf.__main__ import main

if __name__ == "__main__":
    print("Running main function...")
    try:
        main()
        print("Main function completed without errors (unexpected)")
    except Exception as e:
        print(f"Main function raised an exception: {e}")
        print("This means the error handling in __main__.py failed to catch the exception")

    # Restore original sys.exit
    sys.exit = original_exit
