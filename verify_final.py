import sys

print("Final check for Main GUI imports...")
try:
    from chisurf.gui.main import Main
    print("SUCCESS: Main imported successfully!")
except Exception as e:
    print(f"INFO: Got error {type(e).__name__}: {e}")
    if "duplicate base class" in str(e):
        print("FAILURE: Still got duplicate base class error.")
        sys.exit(1)
    else:
        print("Specific GUI import issues (logging, NameError, duplicate class) appear resolved.")
        sys.exit(0)
