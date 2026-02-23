from __future__ import annotations
from chisurf import typing

import abc
import json

import numpy as np
import chinet

import chisurf.base
import chisurf.decorators

T = typing.TypeVar('T', bound='Parameter')


@chisurf.decorators.register
class Parameter(chisurf.base.Base):
    """Scalar parameter backed by a low-level :mod:`chinet` port.

    A :class:`Parameter` represents a single scalar value used in a model
    or fit. The value can be

    - stored directly in an underlying :class:`chinet.Port`,
    - computed dynamically from a Python callable, or
    - linked to another :class:`Parameter`.

    Bounds and a fixed flag are forwarded to the underlying port.

    Examples
    --------
    Create a simple parameter and use it in arithmetic expressions:

    >>> from chisurf.parameter import Parameter
    >>> p = Parameter(name="amp", value=1.5)
    >>> float(p)
    1.5
    >>> q = p + 2.0
    >>> float(q)
    3.5
    """

    @staticmethod
    def check_recursive_link(current, target):
        if id(current) == id(target):
            return True
        if current.link is not None:
            return Parameter.check_recursive_link(current.link, target)
        return False

    @property
    def fit_idx(self):
        import chisurf.fitting
        idxs = chisurf.fitting.find_fit_idx_of_parameter(self)
        if len(idxs) == 0:
            return -1
        if len(idxs) > 1:
            chisurf.logging.warning("Ambiguous link call. Fitting parameter used in multiple fits")
        fit_idx_self = idxs[0]
        return fit_idx_self

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, v: str):
        self._port.name = v
        self._name = v

    @property
    def bounds(self) -> typing.Tuple[float, float]:
        """Lower and upper bounds of the parameter as a 2-tuple.

        The values are stored on the underlying :class:`chinet.Port`.
        """
        return self._port.bounds

    @bounds.setter
    def bounds(self, b: typing.Tuple[float, float]):
        self._port.bounds = np.array(b, dtype=np.float64)

    @property
    def bounds_on(self):
        """Whether bounds are currently enforced on the parameter."""
        return self._port.bounded

    @bounds_on.setter
    def bounds_on(self, v):
        self._port.bounded = bool(v)

    @property
    def value(self) -> float:
        """Current scalar value of the parameter.

        If a callable was passed at construction time, it is evaluated each
        time this property is accessed. The result is clamped to bounds (if
        enabled) and, when the parameter is not fixed, written back to the
        underlying port.

        If the parameter is linked to another :class:`Parameter`, the link
        takes precedence and the callable is ignored.
        """
        # If linked, defer entirely to linked parameter's port value.
        if self.is_linked:
            return self._port.value

        # Compute from callable if available.
        if self._callable:
            try:
                v = float(self._callable())
            except Exception:
                return self._port.value
        else:
            v = float(self._port.value)

        # Apply bounds on read for both callable and non-callable parameters
        # if bounds are enabled. This matches the behaviour expected in the
        # unit tests (``test_bounds``), where reading ``value`` after an
        # out-of-bounds assignment should return the clamped value.
        if self.bounds_on:
            lb, ub = self.bounds
            if np.isfinite(lb):
                v = max(lb, v)
            if np.isfinite(ub):
                v = min(ub, v)

        # Write the clamped value back to the port when the parameter is not
        # fixed, so subsequent reads remain consistent.
        if not self.fixed:
            f = self._port.fixed
            self._port.fixed = False
            self._port.value = v
            self._port.fixed = f

        return v

    @value.setter
    def value(self, value: float):
        """Set the parameter value.

        When the parameter was constructed from a callable, the setter is
        ignored to ensure that the callable remains the single source of
        truth.
        """
        if self._callable:
            return
        f = self._port.fixed
        self._port.fixed = False
        self._port.value = value
        self._port.fixed = f

    @property
    def link(self) -> chisurf.parameter.Parameter:
        return self._link

    @link.setter
    def link(self, link: Parameter|None):
        if isinstance(link, Parameter):
            if Parameter.check_recursive_link(link, self):
                raise ValueError("Cannot create a recursive link between parameters.")
            # This parameter becomes a follower (slave) of the target.
            self._link = link
            self.is_link_master = False
            if self.controller is not None:
                # Followers show the partially-checked link state.
                self.controller.set_linked(True)
            self._port.link = link._port
        elif link is None:
            # Unlink this parameter from any target. The is_link_master flag
            # is *not* modified here so that higher-level helpers (such as
            # fit-group linking) can control master semantics explicitly.
            self._link = None
            self._port.unlink()
            if self.controller is not None:
                self.controller.set_linked(False)

    @property
    def is_linked(self) -> bool:
        """Whether this parameter is linked to another parameter."""
        return self._port.is_linked

    @property
    def fixed(self):
        """Boolean flag indicating whether the parameter is fixed."""
        return self._port.fixed

    @fixed.setter
    def fixed(self, v: bool):
        self._port.fixed = bool(v)

    def __add__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a + b)
        )

    def __mul__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a * b)
        )

    def __truediv__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a / b)
        )

    def __floordiv__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a // b)
        )

    def __sub__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a - b)
        )

    def __mod__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a % b)
        )

    def __pow__(self, other: T) -> T:
        a = self.value
        b = other.value if isinstance(other, Parameter) else other
        return self.__class__(
            value=(a ** b)
        )

    def __invert__(self) -> T:
        a = self.value
        return self.__class__(
            value=(1./a)
        )

    def __float__(self):
        return float(self.value)

    def __eq__(self, other: Parameter) -> bool:
        if isinstance(other, Parameter):
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other: Parameter):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        """Return a compact string representation of the parameter value.

        For integer-like values we avoid a trailing ``.0`` so that tests
        expecting ``"22"`` rather than ``"22.0"`` continue to pass.
        """

        v = float(self.value)
        if v.is_integer():
            return str(int(v))
        return repr(v)

    def __abs__(self):
        return self.__class__(
            value=self.value.__abs__()
        )

    def __getstate__(self):
        d = json.loads(self._port.get_json())
        return {
            'port': d
        }

    def __setstate__(self, state):
        s = json.dumps(state['port'])
        self._port.read_json(s)
        fixed = self._port.fixed
        self._port.fixed = False
        self._port.value = state['port']['value']
        self._port.fixed = fixed

    def __round__(self, n=None):
        return self.__class__(
            value=self.value.__round__()
        )

    @abc.abstractmethod
    def update(self):
        """Hook for subclasses to react to external changes.

        The base :class:`Parameter` does not define an update strategy; this
        method primarily exists so GUI-aware subclasses can synchronize their
        controllers.
        """
        pass

    def __init__(self, value: float = 1.0, link: 'Parameter' = None,
                 lb: float = float("-inf"), ub: float = float("inf"),
                 bounds_on: bool = False, *args, **kwargs):
        """Initialize a :class:`Parameter` instance.

        Parameters
        ----------
        value : float or callable
            Initial value of the parameter, or a callable computing it
            dynamically.
        link : Parameter, optional
            Another parameter this one should be linked to.
        lb, ub : float, optional
            Lower and upper bounds for the value stored on the underlying
            :class:`chinet.Port`.
        bounds_on : bool, optional
            If *True*, the bounds are enforced on the port.
        """
        super().__init__(*args, **kwargs)
        self._name = kwargs.pop('name', '')
        self.is_output = bool(kwargs.pop('is_output', False))
        # Hint for GUIs: parameters that serve as link targets for other
        # parameters within a fit group are marked as "link masters".
        # This is purely a visual/UI role and does not affect the core
        # numerical behaviour of links handled by the underlying port.
        self.is_link_master = bool(kwargs.pop('is_link_master', False))
        # Optional free-form description used by fitting GUIs to show
        # human-readable details for a parameter.
        desc = kwargs.pop('description', "")
        registry_id = kwargs.pop('registry_id', None)
        if not desc:
            try:
                meta = getattr(chisurf.settings, "fitting_parameters", {})
                params_meta = meta.get("parameters", meta) if isinstance(meta, dict) else {}
                entry = None
                if isinstance(params_meta, dict):
                    if registry_id is not None:
                        entry = params_meta.get(registry_id)
                    if entry is None:
                        entry = params_meta.get(self._name)
                    if entry is None:
                        for _key, _val in params_meta.items():
                            if not isinstance(_val, dict):
                                continue
                            aliases = _val.get("aliases") or []
                            if isinstance(aliases, list) and self._name in aliases:
                                entry = _val
                                break
                if isinstance(entry, dict):
                    d_reg = entry.get("description")
                    if isinstance(d_reg, str) and d_reg:
                        desc = d_reg
            except Exception:
                pass
        self.description = desc
        port = kwargs.pop('port', None)
        if port is not None:
            self._port = port
            self._callable = None
        else:
            if callable(value):
                self._callable = value
                self._port = chinet.Port(
                    value=np.atleast_1d(0.0).astype(np.float64),
                    name=self._name, lb=lb, ub=ub, is_bounded=bounds_on
                )
            else:
                self._callable = None
                self._port = chinet.Port(
                    value=np.atleast_1d(value).astype(np.float64),
                    name=self._name, lb=lb, ub=ub, is_bounded=bounds_on
                )
        self._link = link
        if isinstance(link, Parameter):
            self._port.link = link._port
        self.controller = None

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of this parameter's state.

        The state is intentionally lightweight and focuses on the
        high-level attributes expected to round-trip in tests and project
        save/load: ``value``, ``bounds_on``, ``bounds`` and ``fixed``.
        """

        try:
            lb, ub = self.bounds
        except Exception:
            lb, ub = float("-inf"), float("inf")
        try:
            desc = getattr(self, "description", "")
        except Exception:
            desc = ""
        return {
            "value": float(self.value),
            "bounds_on": bool(self.bounds_on),
            "bounds": [float(lb), float(ub)],
            "fixed": bool(self.fixed),
            "description": str(desc),
        }

    def set_state(self, state: dict) -> None:
        """Restore parameter state from :meth:`get_state` output.

        After restoring the core scalar attributes, :meth:`update` is called
        so any attached GUI controller can refresh itself.
        """

        if not isinstance(state, dict):
            return

        try:
            if "bounds" in state:
                b = state["bounds"]
                if isinstance(b, (list, tuple)) and len(b) == 2:
                    self.bounds = (float(b[0]), float(b[1]))
            if "bounds_on" in state:
                self.bounds_on = bool(state["bounds_on"])
            if "fixed" in state:
                self.fixed = bool(state["fixed"])
            if "value" in state:
                self.value = float(state["value"])
            if "description" in state:
                try:
                    self.description = str(state["description"])
                except Exception:
                    pass
        except Exception:
            return

        try:
            self.update()
        except Exception:
            # Parameters without a concrete update hook simply ignore this.
            pass


class ParameterGroup(chisurf.base.Base):
    """Container for a list of :class:`Parameter` objects.

    The group behaves like a light-weight collection that forwards attribute
    access to contained parameters when appropriate. It is mainly used to
    manage related parameters in a convenient way.

    Examples
    --------
    >>> from chisurf.parameter import Parameter, ParameterGroup
    >>> p1 = Parameter(name="a", value=1.0)
    >>> p2 = Parameter(name="b", value=2.0)
    >>> group = ParameterGroup(parameters=[p1, p2])
    >>> group.parameter_names
    ['a', 'b']
    >>> sum(group.values)
    3.0
    """

    def __init__(
            self,
            parameters: typing.List[Parameter] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if parameters is None:
            parameters = list()
        self._parameter = parameters

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of all contained parameters.

        The structure mirrors :meth:`to_dict` but is intended specifically for
        lightweight state transfer and testing.
        """

        try:
            return self.to_dict()
        except Exception:
            return {}

    def set_state(self, state: dict) -> None:
        """Restore group/parameter state from :meth:`get_state` output.

        This forwards to :meth:`from_dict` and then asks each contained
        parameter to :meth:`update`, allowing any associated UI controllers to
        refresh.
        """

        if not isinstance(state, dict):
            return
        try:
            self.from_dict(state)
        except Exception:
            return
        try:
            for p in getattr(self, "parameters", []):
                upd = getattr(p, "update", None)
                if callable(upd):
                    upd()
        except Exception:
            pass

    def __setattr__(
            self,
            k: str,
            v: object
    ):
        try:
            propobj = getattr(self, k, None)
            if isinstance(propobj, property):
                if propobj.fset is None:
                    raise AttributeError("can't set attribute")
                propobj.fset(self, v)
            elif isinstance(propobj, chisurf.parameter.Parameter):
                propobj.value = v
            else:
                super().__setattr__(k, v)
        except KeyError:
            super().__setattr__(k, v)

    def __getattr__(self, key: str):
        v = super().__getattr__(key=key)
        if isinstance(v, chisurf.parameter.Parameter):
            return v.value
        return v

    def append(self,
            parameter: Parameter,
            **kwargs
    ):
        self._parameter.append(parameter)

    def clear(self):
        self._parameter = list()

    @property
    def parameters(self) -> typing.List[Parameter]:
        return self._parameter

    @property
    def parameter_names(self) -> typing.List[str]:
        return [p.name for p in self.parameters]

    @property
    def values(self) -> np.array:
        return [p.value for p in self.parameters]

    # def save_txt(
    #         self,
    #         filename: str,
    #         sep: str = '\t'
    # ):
    #     with open(filename, 'w') as fp:
    #         s = ""
    #         for ph in self.parameter_names:
    #             s += ph + sep
    #         s += "\n"
    #         for l in self.values:
    #             for p in l:
    #                 s += "%.5f%s" % (p, sep)
    #             s += "\n"
    #         fp.write(s)
    #
