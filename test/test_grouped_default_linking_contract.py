from pathlib import Path


def test_grouped_fits_auto_link_non_nuisance_parameters_contract():
    path = Path(__file__).resolve().parents[1] / "chisurf" / "macros" / "core_fit.py"
    src = path.read_text(encoding="utf-8")

    assert "def _auto_link_non_nuisance_group_parameters" in src
    assert "_collect_group_nuisance_parameter_names" in src
    assert "generic\", \"corrections\", \"convolve\"" in src
    assert "linked_masters, linked_followers = _auto_link_non_nuisance_group_parameters(fit_group)" in src
    assert "action_type=\"fit_group_auto_link\"" in src
