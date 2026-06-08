import chisurf as cs
# Test script to verify that a popup message is displayed when a non-existent setup is specified
# This script should be run in the chisurf console

# First, set a valid experiment
cs.current_experiment = 'TCSPC'

# Then, try to set a non-existent setup
# This should display a popup message
cs.current_setup = 'NON_EXISTENT_SETUP'

# The script should continue execution after the popup is closed
print("Script execution continued after popup")