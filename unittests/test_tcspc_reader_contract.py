from pathlib import Path


def _tcspc_source() -> str:
    path = Path(__file__).resolve().parents[1] / "chisurf" / "fio" / "fluorescence" / "tcspc.py"
    return path.read_text(encoding="utf-8")


def test_ex_array_is_initialized_for_datacurve_creation():
    src = _tcspc_source()
    assert "ex = np.zeros(x.shape)" in src
    assert "ex=ex" in src


def test_vm_is_supported_in_polarization_naming():
    src = _tcspc_source()
    assert "'vm'" in src
    assert "polarization.upper()" in src


def test_descriptive_jordi_names_have_vv_vh_labels():
    src = _tcspc_source()
    assert "VV" in src
    assert "VH" in src
