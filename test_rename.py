import os
import sys
import tttrlib
import shutil

original_ptu = r"e:\dev\chisurf\test\data\clsm\Leica_SP8.ptu"
test_ptu = os.path.abspath("test_real_µ_file.ptu")

# copy a real PTU file so we don't mess up the original
shutil.copy(original_ptu, test_ptu)

# now rename it to an ascii name
temp_name = test_ptu.replace("µ", "m")
os.rename(test_ptu, temp_name)

print("Opening", temp_name, "with tttrlib...")
tt = tttrlib.TTTR(temp_name)
print("File opened successfully.")

# try to rename it back
print("Attempting to rename back to", test_ptu)
try:
    os.rename(temp_name, test_ptu)
    print("Rename successful! tttrlib does NOT hold the file open.")
except Exception as e:
    print("Rename failed:", e)

# Clean up
try:
    del tt
except: pass

try:
    os.remove(test_ptu)
except: pass

try:
    os.remove(temp_name)
except: pass

print("Done")
