#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from test_dll_debug import test_conversion_and_spc_file, test_dll_loading, test_function_prototype

def main():
    print("Quick test of batch processing")

    dll = test_dll_loading()
    if dll:
        # Set up function prototypes first
        if test_function_prototype(dll):
            test_conversion_and_spc_file(dll)
        else:
            print("Failed to set up prototypes")
    else:
        print("DLL not found")

if __name__ == "__main__":
    main()
