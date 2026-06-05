import os
import pathlib
import sys

# Add chisurf to path
TOPDIR = pathlib.Path(__file__).parent
sys.path.append(str(TOPDIR))

import chisurf.core.models.tcspc.lifetime
import chisurf.core.models.tcspc.anisotropy
import chisurf.core.fitting.fit

def reproduce_bug():
    print("Reproducing bug...")
    
    # Create Lifestyle fit
    lt = chisurf.core.models.tcspc.lifetime.Lifetime(short='L')
    print(f"Initial n: {lt.n}")
    
    print("Appending component 1...")
    lt.append()
    print(f"After append 1, n: {lt.n}") 
    print(f"Parameters: {list(lt.parameters_all_dict.keys())}") 
    
    print("Popping component...")
    lt.pop()
    print(f"After pop, n: {lt.n}") 
    print(f"Parameters in dict: {list(lt.parameters_all_dict.keys())}") 
    
    print("Appending component again (should use index 1)...")
    lt.append()
    print(f"After append after pop, n: {lt.n}")
    print(f"Parameters: {list(lt.parameters_all_dict.keys())}")
    
    print("Appending component 2 (should use index 2)...")
    lt.append()
    print(f"After second append, n: {lt.n}")
    print(f"Parameters: {list(lt.parameters_all_dict.keys())}")

if __name__ == "__main__":
    reproduce_bug()
