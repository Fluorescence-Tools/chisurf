# Test script to reproduce the crash when changing setup
cs.current_experiment = 'TCSPC'
# Try changing to a valid setup
cs.current_setup = 'TXT/CSV'
print("Setup changed successfully")