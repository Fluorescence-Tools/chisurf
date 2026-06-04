from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass
class DatasetSummary:
    index: int
    uid: str
    name: str
    type: str
    experiment: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert this DTO to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DatasetSummary:
        """Create a DTO from a plain dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary with DTO field values.

        """
        return cls(**{k: v for k, v in d.items() if k in cls._fields()})

    @classmethod
    def _fields(cls) -> set:
        """Return the set of field names declared on this dataclass."""
        return {f.name for f in dataclasses.fields(cls)}


@dataclasses.dataclass
class DatasetDetail(DatasetSummary):
    length: Optional[int] = None


@dataclasses.dataclass
class FitSummary:
    index: int
    uid: str
    name: str
    type: str
    chi2: Optional[float]
    dataset_uid: str
    dataset_name: str
    model_name: str
    parameter_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert this DTO to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FitSummary:
        """Create a DTO from a plain dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary with DTO field values.

        """
        return cls(**{k: v for k, v in d.items() if k in cls._fields()})

    @classmethod
    def _fields(cls) -> set:
        """Return the set of field names declared on this dataclass."""
        return {f.name for f in dataclasses.fields(cls)}


@dataclasses.dataclass
class ParameterDTO:
    name: str
    value: Optional[float]
    fixed: bool
    bounds_on: bool
    bounds: Optional[Tuple[float, float]]
    linked_to: Optional[str]
    error_estimate: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert this DTO to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ParameterDTO:
        """Create a DTO from a plain dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary with DTO field values.

        """
        return cls(**{k: v for k, v in d.items() if k in cls._fields()})

    @classmethod
    def _fields(cls) -> set:
        """Return the set of field names declared on this dataclass."""
        return {f.name for f in dataclasses.fields(cls)}


@dataclasses.dataclass
class FitDetail(FitSummary):
    parameters: List[ParameterDTO] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SetupDTO:
    name: str
    experiment_name: str
    properties: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this DTO to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SetupDTO:
        """Create a DTO from a plain dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary with DTO field values.

        """
        return cls(**{k: v for k, v in d.items() if k in cls._fields()})

    @classmethod
    def _fields(cls) -> set:
        """Return the set of field names declared on this dataclass."""
        return {f.name for f in dataclasses.fields(cls)}


@dataclasses.dataclass
class ProjectInfoDTO:
    project_path: Optional[str]
    project_name: Optional[str]
    fit_count: int
    dataset_count: int
    experiment_names: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this DTO to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProjectInfoDTO:
        """Create a DTO from a plain dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary with DTO field values.

        """
        return cls(**{k: v for k, v in d.items() if k in cls._fields()})

    @classmethod
    def _fields(cls) -> set:
        """Return the set of field names declared on this dataclass."""
        return {f.name for f in dataclasses.fields(cls)}


@dataclasses.dataclass
class ActionResultDTO:
    ok: bool
    message: str = ""
    data: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this DTO to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ActionResultDTO:
        """Create a DTO from a plain dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary with DTO field values.

        """
        return cls(**{k: v for k, v in d.items() if k in cls._fields()})

    @classmethod
    def _fields(cls) -> set:
        """Return the set of field names declared on this dataclass."""
        return {f.name for f in dataclasses.fields(cls)}


def dto_to_json(dto: Any) -> str:
    """Serialize a DTO (or any object) to a JSON string.

    Parameters
    ----------
    dto : object
        Object with an optional ``to_dict()`` method.

    """
    if hasattr(dto, "to_dict"):
        return json.dumps(dto.to_dict())
    return json.dumps(dto)


def to_json_safe(obj: Any) -> Any:
    """Convert a value to a JSON-safe type."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)