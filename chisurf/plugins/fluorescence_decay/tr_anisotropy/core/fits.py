"""Parameter link/constraint plan for the VV/VH anisotropy global fit.

Creating an anisotropy fit means building a VV and a VH lifetime fit, wrapping
them in a global fit and then wiring a specific set of parameter *links* and
*constraints* between them (the VH channel borrows VV's photon count, rotation
components, lifetimes and instrument-correction factors). That wiring is long,
order-sensitive and easy to get wrong, so it is expressed here as an ordered,
Qt-free list of operations that :func:`apply_link_plan` replays against a fitting
client. Encoding it as data makes it unit-testable without a live fit.
"""

from __future__ import annotations

import typing


def build_link_plan(
    n_lifetime: int,
    n_rotation: int,
    corrections: typing.Mapping[str, float],
) -> list[dict]:
    """Return the ordered VV↔VH link/constraint operations.

    Parameters
    ----------
    n_lifetime : int
        Number of lifetime components.
    n_rotation : int
        Number of rotation (anisotropy) components.
    corrections : mapping
        Instrument corrections with keys ``"g_factor"``, ``"l1"`` and ``"l2"``.

    Returns
    -------
    list of dict
        Operations, each a dict with an ``"op"`` key — one of ``"link"``,
        ``"set_value"``, ``"set_fixed"`` or ``"update"`` (see
        :func:`apply_link_plan` for their semantics).
    """
    g = float(corrections.get("g_factor", 1.0))
    l1 = float(corrections.get("l1", 0.0))
    l2 = float(corrections.get("l2", 0.0))

    plan: list[dict] = []

    # photon count shared, both released
    plan.append({"op": "link", "name": "n0", "target": "n0"})
    plan.append({"op": "set_fixed", "name": "n0", "fixed": False, "fit": "vv"})
    plan.append({"op": "set_fixed", "name": "n0", "fixed": False, "fit": "vh"})

    # instrument corrections: set on VV then fix
    for name, value in (("l1", l1), ("l2", l2), ("g", g)):
        plan.append({"op": "set_fixed", "name": name, "fixed": False, "fit": "vv"})
        plan.append({"op": "set_value", "name": name, "value": value, "fit": "vv"})
        plan.append({"op": "set_fixed", "name": name, "fixed": True, "fit": "vv"})
        plan.append({"op": "update", "fit": "vv"})

    # rotation components: link amplitude b(i) and correlation time rho(i)
    for i in range(1, n_rotation + 1):
        plan.append({"op": "link", "name": f"rho({i})", "target": f"rho({i})"})
        plan.append({"op": "link", "name": f"b({i})", "target": f"b({i})"})

    # lifetime components: link fraction xL{i} and lifetime tL{i}
    for i in range(1, n_lifetime + 1):
        plan.append({"op": "link", "name": f"xL{i}", "target": f"xL{i}"})
        plan.append({"op": "link", "name": f"tL{i}", "target": f"tL{i}"})

    # background fixed on both channels
    plan.append({"op": "set_fixed", "name": "lb", "fixed": True, "fit": "vv"})
    plan.append({"op": "set_fixed", "name": "lb", "fixed": True, "fit": "vh"})

    # corrections stay fixed on VV and are linked into VH
    for name in ("l1", "l2", "g"):
        plan.append({"op": "set_fixed", "name": name, "fixed": True, "fit": "vv"})
    for name in ("l1", "l2", "g"):
        plan.append({"op": "link", "name": name, "target": name})

    return plan


def apply_link_plan(
    fit_client: typing.Any,
    plan: typing.Sequence[dict],
    vv_index: int,
    vh_index: int,
) -> None:
    """Replay a :func:`build_link_plan` plan against *fit_client*.

    Operation semantics (``fit`` selects VV or VH by index; links always point
    the VH parameter at the matching VV parameter):

    * ``link`` — ``fit_client.link_parameters(name, target, vh_index, vv_index)``
    * ``set_value`` — ``fit_client.set_parameter_value(name, value, fit_index)``
    * ``set_fixed`` — ``fit_client.set_parameter_fixed(name, fixed, fit_index)``
    * ``update`` — ``fit_client.update_fit(fit_index)``
    """

    def _index(which: str) -> int:
        return vv_index if which == "vv" else vh_index

    for op in plan:
        kind = op["op"]
        if kind == "link":
            fit_client.link_parameters(
                parameter_name=op["name"],
                target_parameter_name=op["target"],
                fit_index=vh_index,
                target_fit_index=vv_index,
            )
        elif kind == "set_value":
            fit_client.set_parameter_value(
                parameter_name=op["name"], value=op["value"], fit_index=_index(op["fit"])
            )
        elif kind == "set_fixed":
            fit_client.set_parameter_fixed(
                parameter_name=op["name"], fixed=op["fixed"], fit_index=_index(op["fit"])
            )
        elif kind == "update":
            fit_client.update_fit(fit_index=_index(op["fit"]))
