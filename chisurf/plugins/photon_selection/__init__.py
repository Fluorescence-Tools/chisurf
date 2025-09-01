"""
Photon/Burst Selection

This plugin provides a graphical interface for selecting and filtering photon data 
from Time-Tagged Time-Resolved (TTTR) measurements.

Features:
- Selection of detector channels for analysis
- Filtering of photon events based on detector parameters
- Interactive wizard interface for intuitive data selection
- Seamless integration with TTTR data processing workflows

The plugin is essential for preprocessing single-molecule fluorescence data, 
allowing users to isolate specific photon populations before further analysis 
such as correlation, burst analysis, or intensity trace examination.

The selection process uses a two-step wizard:
1. First page: Select detector channels of interest
2. Second page: Apply filters to the photon data based on selected detectors

Once the selection is complete, the filtered photon data can be used in 
subsequent analysis steps within the ChiSurf application.
"""

name = "TTTR:Photon/Burst Selection"
