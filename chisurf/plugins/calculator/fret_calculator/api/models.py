"""JSON-serializable DTOs for the FRET Calculator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FretSettings:
    """Input parameters for heteroFRET computation.

    Attributes
    ----------
    R : float
        Donor-acceptor distance.
    R0 : float
        Förster radius.
    tau0 : float
        Donor lifetime without FRET.
    kappa2 : float
        Orientation factor.
    sigma : float
        Width of Gaussian distance distribution.
    """

    R: float = 50.0
    R0: float = 52.0
    tau0: float = 4.0
    kappa2: float = 0.667
    sigma: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FretSettings:
        """Construct from a JSON-compatible dictionary."""
        return cls(
            R=float(data.get("R", 50.0)),
            R0=float(data.get("R0", 52.0)),
            tau0=float(data.get("tau0", 4.0)),
            kappa2=float(data.get("kappa2", 0.667)),
            sigma=float(data.get("sigma", 0.0)),
        )


@dataclass
class FretResult:
    """Output of heteroFRET computation.

    Attributes
    ----------
    R : float
        Donor-acceptor distance.
    R0 : float
        Förster radius.
    tau0 : float
        Donor lifetime without FRET.
    kappa2 : float
        Orientation factor.
    sigma : float
        Width of Gaussian distance distribution.
    E : float
        FRET efficiency.
    tau_DA : float
        Donor lifetime with acceptor.
    kFRET : float
        FRET rate constant.
    """

    R: float = 0.0
    R0: float = 0.0
    tau0: float = 0.0
    kappa2: float = 0.0
    sigma: float = 0.0
    E: float = 0.0
    tau_DA: float = 0.0
    kFRET: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FretResult:
        """Construct from a JSON-compatible dictionary."""
        return cls(**{k: float(data.get(k, 0.0)) for k in cls.__dataclass_fields__})


@dataclass
class HomoFretSettings:
    """Input parameters for homoFRET computation.

    Attributes
    ----------
    t_RM : float
        Anisotropy relaxation time.
    rho : float
        Rotational correlation time without homoFRET.
    tau0 : float
        Donor fluorescence lifetime.
    R0 : float
        Förster radius.
    """

    t_RM: float = 1.0
    rho: float = 2.0
    tau0: float = 4.0
    R0: float = 52.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HomoFretSettings:
        """Construct from a JSON-compatible dictionary."""
        return cls(
            t_RM=float(data.get("t_RM", 1.0)),
            rho=float(data.get("rho", 2.0)),
            tau0=float(data.get("tau0", 4.0)),
            R0=float(data.get("R0", 52.0)),
        )


@dataclass
class HomoFretResult:
    """Output of homoFRET computation.

    Attributes
    ----------
    k_homo : float
        homoFRET exchange rate.
    R_DA : float
        Effective donor-acceptor distance.
    t_RM : float
        Anisotropy relaxation time.
    rho : float
        Rotational correlation time without homoFRET.
    tau0 : float
        Donor fluorescence lifetime.
    R0 : float
        Förster radius.
    """

    k_homo: float = 0.0
    R_DA: float = 0.0
    t_RM: float = 0.0
    rho: float = 0.0
    tau0: float = 0.0
    R0: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HomoFretResult:
        """Construct from a JSON-compatible dictionary."""
        return cls(**{k: float(data.get(k, 0.0)) for k in cls.__dataclass_fields__})
