import numpy as np
import chisurf.models
from chisurf.fitting.parameter import FittingParameter
from chisurf.plugins.kappa2_dist.k2dfun import kappasq, s2delta

class AnisotropyToKappaModel(chisurf.fitting.parameter.FittingParameterGroup):
    """Calculate the orientation factor kappa^2 from anisotropy values.

    This model calculates the orientation factor kappa^2 for a Donor-Acceptor
    system assuming a wobbling-in-a-cone model, based on the residual
    anisotropies of the donor, acceptor, and FRET-sensitized emission.
    """

    def __init__(self, fit=None, r0=0.38, **kwargs):
        super().__init__(name="AnisotropyToKappa", fit=fit, **kwargs)
        self.r0 = r0
        
        # Input parameters
        self.r_donor = FittingParameter(name="r_donor", value=0.1, bounds=(0.0, r0))
        self.r_acceptor = FittingParameter(name="r_acceptor", value=0.1, bounds=(0.0, r0))
        self.r_sensitized = FittingParameter(name="r_sensitized", value=0.1, bounds=(-r0/2.0, r0))
        
        # Output parameter (calculated)
        self.kappa_squared = FittingParameter(name="kappa_squared", value=2.0/3.0, fixed=True)
        
        # Add to group
        self.append(self.r_donor)
        self.append(self.r_acceptor)
        self.append(self.r_sensitized)
        self.append(self.kappa_squared)

    def update_model(self):
        """Update the calculated kappa^2 value based on current anisotropy values."""
        try:
            # S2 = +/- sqrt(r_inf / r0)
            # Standard convention for donor and acceptor
            s2_donor = -np.sqrt(max(0.0, self.r_donor.value / self.r0))
            s2_acceptor = np.sqrt(max(0.0, self.r_acceptor.value / self.r0))
            
            # Calculate delta angle between symmetry axes
            _, delta = s2delta(
                s2_donor=s2_donor,
                s2_acceptor=s2_acceptor,
                r_inf_AD=self.r_sensitized.value,
                r_0=self.r0
            )
            
            # Calculate kappa^2 for the average orientation in the WIC model
            # Assuming beta1 = beta2 = 0 for the symmetry axes
            self.kappa_squared.value = kappasq(
                delta=delta,
                sD2=s2_donor,
                sA2=s2_acceptor,
                beta1=0.0,
                beta2=0.0
            )
        except Exception:
            self.kappa_squared.value = 2.0/3.0

    def __str__(self):
        s = f"AnisotropyToKappaModel (r0={self.r0:.3f})\n"
        s += f"  r_donor:      {self.r_donor.value:.4f}\n"
        s += f"  r_acceptor:   {self.r_acceptor.value:.4f}\n"
        s += f"  r_sensitized: {self.r_sensitized.value:.4f}\n"
        s += f"  kappa_squared: {self.kappa_squared.value:.4f}"
        return s
