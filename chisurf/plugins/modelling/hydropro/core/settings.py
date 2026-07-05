"""HYDROPRO / HYDRO++ calculation settings (pure data, no Qt)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass
class HydroProSettings:
    """Hydrodynamic calculation parameters for HYDROPRO / HYDRO++.

    Field names and defaults follow the HYDROPRO 10 manual (§3). Values are
    written verbatim into the ``hydropro.dat`` main input file.
    """

    indmode: int = 1            # 1 atomic/shell, 2 residue/shell, 4 residue/bead
    aer: float = 2.9            # AER per manual (Å)
    nsig: int = 6               # number of sigma values; -1 for automatic
    sigmin: float = 1.0         # only when nsig != -1
    sigmax: float = 2.0         # only when nsig != -1
    t: float = 20.0             # temperature (°C)
    eta: float = 0.01           # solvent viscosity (poise)
    rm: float = 100000.0        # molecular weight (Da)
    vbar: float = 0.74          # partial specific volume (cm^3/g)
    rho: float = 1.0            # solution density (g/cm^3)
    nq: int = -1                # scattering values; 0 omit, -1 automatic, >0 provide qmax
    qmax: float = 0.0           # only when nq > 0
    ns: int = -1                # distance distribution intervals; 0 omit, -1 automatic, >0 provide rmax
    rmax: float = 0.0           # only when ns > 0
    ntrials: int = 0            # Monte Carlo covolume trials
    idif: int = 1               # 1 to output full diffusion tensors

    def validate(self) -> None:
        """Raise :class:`ValueError` if the parameter combination is invalid."""
        if self.indmode not in (1, 2, 4):
            raise ValueError("INDMODE must be 1, 2, or 4")
        if self.indmode in (1, 2) and self.nsig != -1 and self.nsig <= 2:
            raise ValueError("NSIG must be > 2 or -1 (automatic) in shell modes")
        if self.nq > 0 and self.qmax <= 0:
            raise ValueError("QMAX must be > 0 when NQ > 0")
        if self.ns > 0 and self.rmax <= 0:
            raise ValueError("RMAX must be > 0 when NS > 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HydroProSettings":
        """Build settings from a mapping, coercing each field to its type."""
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data or data[f.name] is None:
                continue
            value = data[f.name]
            try:
                if f.type == "int" or isinstance(getattr(cls(), f.name), int):
                    kwargs[f.name] = int(float(value))
                else:
                    kwargs[f.name] = float(value)
            except (TypeError, ValueError):
                continue
        return cls(**kwargs)


__all__ = ["HydroProSettings"]
