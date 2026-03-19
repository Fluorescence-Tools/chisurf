import os
import tttrlib

# Create a mock TTTR file with a unicode name
filename = "test_µ_file.ptu"
with open(filename, "w") as f:
    f.write("mock")

try:
    print(f"Trying to open string path: {filename}")
    tt = tttrlib.TTTR(filename)
    print("Success with string path")
except Exception as e:
    print(f"Failed with string path: {e}")

try:
    print(f"Trying to open unicode path: {filename}")
    tt = tttrlib.TTTR(filename)
    print("Success with unicode")
except Exception as e:
    print(f"Failed with unicode: {e}")

try:
    print("Trying to open utf-8 encoded path")
    tt = tttrlib.TTTR(filename.encode('utf-8'))
    print("Success with utf-8 encoded bytes")
except Exception as e:
    print(f"Failed with utf-8 encoded bytes: {e}")

print("Done")
