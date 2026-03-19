from pathlib import Path


def test_irf_is_normalized_before_convolution_paths():
    path = Path(__file__).resolve().parents[1] / "chisurf" / "models" / "tcspc" / "nusiance.py"
    src = path.read_text(encoding="utf-8")

    norm_idx = src.find("irf_y = irf_y / np.sum(irf_y)")
    assert norm_idx != -1

    periodic_idx = src.find("convolve_lifetime_spectrum_periodic_nb", norm_idx)
    exp_idx = src.find("convolve_lifetime_spectrum_nb", norm_idx)
    full_idx = src.find("np.convolve(data, irf_y", norm_idx)

    assert periodic_idx > norm_idx
    assert exp_idx > norm_idx
    assert full_idx > norm_idx
