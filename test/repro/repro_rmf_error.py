import sys
from pathlib import Path
import numpy as np

# Add the project root to sys.path
sys.path.append(str(Path(r"E:\dev\chisurf")))

from chisurf.plugins.chimol.chimol.io.rmf import load_rmf_full

def verify():
    path = Path(r"E:\dev\chisurf\chisurf\plugins\chimol\tests\data\0.rmf3")
    print(f"Loading {path}...")
    data = load_rmf_full(path)
    
    frames = data["frames"]
    radii = data["radii"]
    hierarchy = data["hierarchy"]
    restraints = data.get("restraints", [])
    bond_pairs = data.get("bond_pairs")
    
    print(f"Number of frames: {len(frames)}")
    print(f"Number of particles: {frames.shape[1]}")
    print(f"Number of restraints: {len(restraints)}")
    print(f"Number of bond pairs: {len(bond_pairs) if bond_pairs is not None else 0}")
    
    # Check for non-zero coordinates
    if np.all(frames == 0):
        print("ERROR: All coordinates are zero!")
        sys.exit(1)
        
    # Check for non-zero radii
    if np.all(radii <= 0):
        print("ERROR: All radii are <= 0!")
        sys.exit(1)
        
    print("SUCCESS!")

if __name__ == "__main__":
    verify()
