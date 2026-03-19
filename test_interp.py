import sys
sys.path.insert(0, 'e:/dev/chisurf')
import numpy as np

from chisurf.plugins._dev.spectra_viewer.simulator.crosstalk import interp

print("Testing discrete interp")
res = interp(("discrete", [488.0, 561.0]))
print("Result shape:", res.shape)
print("Non-zero elements:", np.count_nonzero(res))
print("Done")
