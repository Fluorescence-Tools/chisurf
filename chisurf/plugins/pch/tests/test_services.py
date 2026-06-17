from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState
from chisurf.plugins.pch.backend.services import register_services


def test_register_services():
    state = SessionState()
    dispatcher = ServiceDispatcher(state)
    register_services(dispatcher)
    assert dispatcher.has_method("pch.load_tttr")
    assert dispatcher.has_method("pch.compute")
    assert dispatcher.has_method("pch.fit")


def test_fit_handler_simple():
    state = SessionState()
    dispatcher = ServiceDispatcher(state)
    register_services(dispatcher)

    k_vals = list(range(10))
    p_exp = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.001]

    result = dispatcher.dispatch(
        "pch.fit",
        {
            "k_vals": k_vals,
            "p_exp": p_exp,
            "n_components": 1,
            "initial_epsilons": [2.0],
            "initial_Ns": [3.0],
            "fit_low": 0,
            "fit_high": 9,
        },
    )
    assert result.get("ok", True)
    r = result.get("result", {})
    assert "epsilons" in r
    assert "avg_Ns" in r
    assert len(r["epsilons"]) == 1
    assert r["chi2"] >= 0
