import chisurf as cs
# Test script to verify that changing to an FCS setup doesn't crash
# This script should be run in the chisurf console

# First, set the experiment to FCS
cs.current_experiment = 'FCS'

# Then, try to set a setup
# This should not crash
cs.current_setup = 'CSV'

# Print a message to confirm that the script completed successfully
print("FCS setup changed successfully")