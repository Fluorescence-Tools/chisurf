from __future__ import annotations
from chisurf import typing

import numpy as np
import chinet as cn
from typing import TYPE_CHECKING

import chisurf.decorators
import chisurf.parameter

from chisurf.curve import Curve
from chisurf.models import model
from chisurf.models.parameter_transform import ParameterTransformModel

if TYPE_CHECKING:
    from chisurf.fitting.fit import Fit, FitGroup

