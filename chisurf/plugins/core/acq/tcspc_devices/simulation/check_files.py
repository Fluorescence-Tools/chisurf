import os
import glob

files = sorted(glob.glob('*.spc'))
for f in files:
    size = os.path.getsize(f)
    print(f'{f}: {size} bytes')
