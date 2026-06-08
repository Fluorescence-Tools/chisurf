import chisurf as cs
# Test script to update the UI when properties of current_setup are changed
cs.current_experiment = 'TCSPC'
cs.current_setup = 'TXT/CSV'
cs.current_setup.is_jordi = True
cs.current_setup.use_header = False
cs.current_setup.matrix_columns = []
cs.current_setup.g_factor = 0.950000
cs.current_setup.polarization = 'vm'
cs.current_setup.rep_rate = 20.0
cs.current_setup.rebin = (1, 1)
cs.current_setup.dt = 0.02
# Update the UI to reflect the changes
cs.update_setup_ui()