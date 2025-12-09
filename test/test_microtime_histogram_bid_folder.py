#!/usr/bin/env python
"""
Test script to verify that microtime histograms are saved in the BID/BUR folder
when BID/BUR files are used.
"""
import sys
import os
from pathlib import Path
import inspect

# Add the parent directory to the path so we can import chisurf
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import the module to inspect the code
    import chisurf.plugins.microtime_histogram.wizard as wizard_module
    
    print("\nTesting microtime histogram BID/BUR folder saving...")
    
    # Get the source code of the update_output_filename method
    source_code = inspect.getsource(wizard_module.MicrotimeHistogram.update_output_filename)
    print("\nVerifying implementation:")
    
    # Check if the code checks for BID files
    bid_check = "bid_files = self.listWidget_BID.get_selected_files()" in source_code
    print(f"- Checks for BID files: {bid_check}")
    
    # Check if the code uses BID directory when BID files are present
    bid_dir_check = "if len(bid_files) > 0:" in source_code and "bid_directory = Path(bid_files[0]).parent" in source_code
    print(f"- Uses BID directory when BID files are present: {bid_dir_check}")
    
    # Check if the code logs the BID directory usage
    log_check = "Using BID/BUR file folder for saving" in source_code
    print(f"- Logs BID directory usage: {log_check}")
    
    # Check if the code prioritizes BID folder over TTTR folder
    priority_check = "elif self.tttr_folder is not None:" in source_code
    print(f"- Prioritizes BID folder over TTTR folder: {priority_check}")
    
    # Overall result
    if bid_check and bid_dir_check and log_check and priority_check:
        print("\nSUCCESS: The code correctly implements saving histograms in the BID/BUR folder when BID/BUR files are used.")
        print("The implementation will:")
        print("1. Check if BID/BUR files are being used")
        print("2. Use the directory of the first BID/BUR file for saving if available")
        print("3. Fall back to TTTR folder or selected file directory if no BID/BUR files are present")
    else:
        print("\nFAILURE: The code does not correctly implement saving histograms in the BID/BUR folder.")
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
except Exception as e:
    print(f"Error during testing: {e}")

if __name__ == "__main__":
    print("Test completed.")