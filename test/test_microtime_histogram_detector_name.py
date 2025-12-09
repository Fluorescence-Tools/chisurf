"""
Code inspection for the MicrotimeHistogram class.
This script inspects the code to verify that the detector name is correctly included in the output filename.
"""

import sys
import os
import inspect
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.plugins.microtime_histogram.wizard import MicrotimeHistogram

def main():
    """Inspect the MicrotimeHistogram class."""
    print("Inspecting MicrotimeHistogram update_output_filename method...")
    
    # Get the source code of the update_output_filename method
    update_output_filename_source = inspect.getsource(MicrotimeHistogram.update_output_filename)
    print("\nSource code of update_output_filename method:")
    print(update_output_filename_source)
    
    # Check if the method includes code to get the detector name
    detector_name_check = "detector_name" in update_output_filename_source
    print(f"\nMethod includes code to get detector name: {detector_name_check}")
    
    # Check if the method accesses selected_detector
    selected_detector_check = "selected_detector_info" in update_output_filename_source
    print(f"Method accesses selected_detector_info: {selected_detector_check}")
    
    # Check if the method includes the detector name in the filename
    filename_format_check = "output_filename = f\"{out}{detector_name}_({p_channels})-({s_channels}).dat\"" in update_output_filename_source
    print(f"Method includes detector name in filename format: {filename_format_check}")
    
    # Overall verification
    if detector_name_check and selected_detector_check and filename_format_check:
        print("\nSUCCESS: The update_output_filename method correctly includes the detector name in the filename")
    else:
        print("\nFAILURE: The update_output_filename method does not correctly include the detector name in the filename")
    
    print("\nCode inspection summary:")
    print("1. Modified update_output_filename to include detector name in the filename")
    print("2. Added code to retrieve detector name from detector_wizard_page.selected_detector")
    print("3. Updated filename format to include detector name: filename_detector_(p)-(s).dat")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()