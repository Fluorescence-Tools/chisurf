from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RicsSettings:
    x_range: Optional[Tuple[int, int]] = None
    y_range: Optional[Tuple[int, int]] = None
    frames_index_pairs: Optional[List[Tuple[int, int]]] = None
    subtract_average: str = "stack"

    # Optional metadata (pixel size, times etc.)
    pixel_duration_us: Optional[float] = None
    line_duration_ms: Optional[float] = None
    pixel_size_nm: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RicsData:
    ics_stack: np.ndarray
    ics_mean: np.ndarray
    ics_std: np.ndarray
    line_shift: np.ndarray
    pixel_shift: np.ndarray

    mask: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.ics_mean.shape)  # type: ignore[return-value]
