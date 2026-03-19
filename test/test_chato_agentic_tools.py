from chisurf.plugins._dev.chato.backend import agentic
from chisurf.plugins._dev.chato.backend.langchain import _tcspc_wizard_intent


def test_make_tools_includes_general_core_tools():
    tools = agentic._make_tcspc_tools("unit-test")
    assert "core_action_catalog" in tools
    assert "core_describe_state" in tools
    assert "core_execute_action" in tools
    assert "core_execute_plan" in tools


def test_tool_help_mentions_general_core_tools():
    txt = agentic._tool_help_text()
    assert "core_action_catalog" in txt
    assert "core_describe_state" in txt
    assert "core_execute_action" in txt
    assert "core_execute_plan" in txt


def test_tcspc_wizard_intent_is_not_generic_fit_trigger():
    assert _tcspc_wizard_intent("/fit this fcs data") is False
    assert _tcspc_wizard_intent("help me fit all fcs curves") is False
    assert _tcspc_wizard_intent("/agent load folder and fit") is False


def test_tcspc_wizard_intent_for_explicit_tcspc_context():
    assert _tcspc_wizard_intent("/tcspc") is True
    assert _tcspc_wizard_intent("/fit tcspc with irf") is True
    assert _tcspc_wizard_intent("help me fit tcspc lifetimes") is True
