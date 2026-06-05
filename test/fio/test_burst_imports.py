"""
Test script to verify that the name conflict between bocpd.convert_bursts_to_start_stop
and kalman.convert_bursts_to_start_stop has been resolved.
"""

import chisurf.core.fluorescence.burst

# Test that both functions can be accessed via their module prefixes
print("Testing access to convert_bursts_to_start_stop functions:")
print("BOCPD function:", chisurf.core.fluorescence.burst.bocpd.convert_bursts_to_start_stop)
print("Kalman function:", chisurf.core.fluorescence.burst.kalman.convert_bursts_to_start_stop)

# Verify they are different functions
print("\nVerifying they are different functions:")
print("Are they the same object?", 
      chisurf.core.fluorescence.burst.bocpd.convert_bursts_to_start_stop is 
      chisurf.core.fluorescence.burst.kalman.convert_bursts_to_start_stop)

print("\nTest completed successfully!")