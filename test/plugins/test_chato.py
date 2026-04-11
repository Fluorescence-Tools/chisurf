from __future__ import annotations

# Consolidated test file: test_chato.py


# --- FROM test_chato_agentic_tools.py ---
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

# --- FROM test_chato_direct_fcs_paths.py ---

from pathlib import Path

from chisurf.plugins._dev.chato.frontend.dock import ChatoDock


def test_extract_first_path_like_windows_file_and_folder_paths():
    file_cmd = 'fit this fcs data: "E:\\dev\\chisurf\\test\\data\\fcs\\kristine\\Kristine_without_error.cor"'
    folder_cmd = "Read the FCS curves in E:\\dev\\chisurf\\test\\data\\fcs\\kristine and make a global fit"

    file_path = ChatoDock._extract_first_path_like(file_cmd)
    folder_path = ChatoDock._extract_first_path_like(folder_cmd)

    assert file_path.endswith(r"test\data\fcs\kristine\Kristine_without_error.cor")
    assert folder_path.endswith(r"test\data\fcs\kristine")


def test_resolve_fcs_curve_paths_supports_folder_inputs():
    base = Path(__file__).resolve().parent / "data" / "fcs" / "kristine"
    paths = ChatoDock._resolve_fcs_curve_paths(str(base))

    assert len(paths) == 2
    assert all(p.suffix.lower() == ".cor" for p in paths)


def test_linkable_global_fcs_parameter_names_prefers_td_and_shape():
    class _Model:
        parameters_all_dict = {"N": 1, "td": 1, "td2": 2, "s": 3, "bt1": 4}

    class _LocalFit:
        model = _Model()

    class _FitGroup:
        grouped_fits = [_LocalFit()]

    names = ChatoDock._linkable_global_fcs_parameter_names(_FitGroup())
    assert names == ["td", "td2", "s"]
