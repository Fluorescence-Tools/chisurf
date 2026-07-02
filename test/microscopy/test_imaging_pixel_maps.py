"""Headless tests for per-pixel N&B + phasor imaging (core + AutoForm tools).

The N&B moment math and HDF5 round-trip run with no data. The phasor and
view-model "run" paths are data-gated on a real TTTR imaging file.
"""
from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import pytest

_HT3 = next(
    iter(glob.glob("/Users/tpeulen/dev/tttr-data/imaging/pq/ht3/pq_ht3_clsm.ht3")),
    None,
)
_needs_data = pytest.mark.skipif(_HT3 is None, reason="TTTR imaging test data not present")


# --- core N&B math (no data) ----------------------------------------------
def test_nb_maps_poisson_recovers_unit_brightness():
    """Pure-Poisson frames have apparent brightness B≈1 (epsilon≈0), N≈mean²/var."""
    from chisurf.core.fluorescence.imaging import nb_maps

    rng = np.random.default_rng(0)
    stack = rng.poisson(lam=25.0, size=(400, 6, 6)).astype(float)
    m = nb_maps(stack)
    assert abs(float(np.mean(m["mean"])) - 25.0) < 1.0
    assert abs(float(np.mean(m["B"])) - 1.0) < 0.05
    assert abs(float(np.mean(m["epsilon"]))) < 0.05
    assert np.all(np.isfinite(m["N"]))


def test_nb_maps_bright_species_has_excess_brightness():
    """A super-Poissonian signal (bright molecules) gives B>1 (epsilon>0)."""
    from chisurf.core.fluorescence.imaging import nb_maps

    rng = np.random.default_rng(1)
    # Each frame: a few bright molecules of brightness eps -> variance = eps*mean extra.
    eps = 5.0
    counts = rng.poisson(lam=4.0, size=(500, 4, 4)) * eps  # scaled counts -> var = eps²·λ, mean = eps·λ
    m = nb_maps(counts.astype(float))
    assert float(np.mean(m["B"])) > 1.5  # clearly super-Poissonian
    assert float(np.mean(m["epsilon"])) > 0.0


def test_nb_maps_single_frame_is_safe():
    """A single frame has undefined variance -> zeros, no crash."""
    from chisurf.core.fluorescence.imaging import nb_maps

    m = nb_maps(np.ones((5, 5)))
    assert m["variance"].shape == (5, 5)
    assert np.all(m["variance"] == 0.0)


def test_maps_to_dataframe_hdf5_roundtrip_and_ndxplorer():
    """maps_to_dataframe -> imaging HDF5 round-trips and loads via ndxplorer."""
    import pandas as pd

    from chisurf.core.fluorescence.imaging import (
        maps_to_dataframe,
        nb_maps,
        write_imaging_hdf5,
    )

    rng = np.random.default_rng(2)
    m = nb_maps(rng.poisson(10.0, size=(50, 8, 8)).astype(float))
    df = maps_to_dataframe(m)
    # Standard imaging layout (matches pixel-wise MLE): Y pixel, X pixel, Pixel Number.
    assert list(df.columns[:3]) == ["Y pixel", "X pixel", "Pixel Number"]
    assert len(df) == 64
    path = os.path.join(tempfile.gettempdir(), "chisurf_nb_roundtrip.h5")
    write_imaging_hdf5(df, path)
    back = pd.read_hdf(path, key="results")
    assert len(back) == 64 and "B" in back.columns
    pytest.importorskip("ndxplorer")
    from ndxplorer import reader as ndx_reader

    ds = ndx_reader.read_mfd_hdf5([path])
    assert ds is not None


def test_add_maps_to_hdf5_enriches_existing_table_and_keeps_source():
    """N&B / phasor add columns to an existing results table, preserving source."""
    import pandas as pd

    from chisurf.core.fluorescence.imaging import (
        add_maps_to_hdf5,
        intensity_maps,
        maps_to_dataframe,
        read_imaging_source,
        write_imaging_hdf5,
    )

    rng = np.random.default_rng(3)
    stack = rng.poisson(12.0, size=(30, 5, 5)).astype(float)
    # Base "imaging HDF5" (intensity + source back-reference).
    path = os.path.join(tempfile.gettempdir(), "chisurf_enrich.h5")
    write_imaging_hdf5(maps_to_dataframe(intensity_maps(stack)), path, source="/data/raw.ht3")
    assert read_imaging_source(path) == "/data/raw.ht3"

    # Enrich with N&B fields -> columns grow, rows unchanged, source preserved.
    from chisurf.core.fluorescence.imaging import nb_maps

    m = nb_maps(stack)
    added = add_maps_to_hdf5(path, {k: m[k] for k in ("N", "B", "epsilon")})
    assert set(added) == {"N", "B", "epsilon"}
    df = pd.read_hdf(path, key="results")
    assert {"intensity", "N", "B", "epsilon"}.issubset(df.columns)
    assert len(df) == 25
    assert read_imaging_source(path) == "/data/raw.ht3"


# --- phasor core (data-gated) ---------------------------------------------
@_needs_data
def test_phasor_maps_land_on_universal_circle():
    """Per-pixel phasor coordinates of real FLIM data fall on the universal circle."""
    import tttrlib

    from chisurf.core.fluorescence.imaging import build_clsm, phasor_maps

    tttr = tttrlib.TTTR(_HT3)
    clsm = build_clsm(tttr, channels=[0])
    maps = phasor_maps(clsm, tttr, frequency=-1.0, n_ph_min=5)
    g, s, n = maps["g"], maps["s"], maps["n_photons"]
    assert g.shape == s.shape and g.ndim == 2
    valid = n > 0
    inside = (g[valid] ** 2 + s[valid] ** 2) <= 1.05
    assert float(np.mean(inside)) > 0.7  # most valid pixels on/within the circle


# --- mean micro-time core (data-gated) ------------------------------------
@_needs_data
def test_mean_micro_time_window_kind_is_positive_ns():
    """The ``mean_micro_time`` window kind yields a per-pixel arrival-time map (ns)."""
    from chisurf.core.fluorescence.imaging import compute_windows

    windows = {"green": {"chs": [0], "ch_p": [], "ch_s": [], "micro_time_ranges": []}}
    results = compute_windows(_HT3, windows, "mean_micro_time", {"n_ph_min": 2})
    wm = results["green"]
    mt, intensity = wm["mean_micro_time"], wm["intensity"]
    assert mt.shape == intensity.shape and mt.ndim == 2
    bright = mt[intensity >= 2]
    # nanosecond-scale, finite, and non-negative on pixels with enough photons.
    assert np.all(np.isfinite(mt))
    assert float(np.nanmax(mt)) < 1e3
    assert float(np.mean(bright > 0)) > 0.5


# --- AutoForm view-models render + run (data-gated for run) ---------------
def test_view_models_build_via_autoform(qtbot):
    """All imaging tools render through AutoForm (view.json resolves)."""
    from chisurf.gui.autoform import AutoForm
    from chisurf.plugins.microscopy.img_pixel_intensity.gui.view_model import (
        IntensityViewModel,
    )
    from chisurf.plugins.microscopy.img_pixel_micro_time.gui.view_model import (
        MicroTimeViewModel,
    )
    from chisurf.plugins.microscopy.img_pixel_nb.gui.view_model import NBViewModel
    from chisurf.plugins.microscopy.img_pixel_phasor.gui.view_model import (
        PhasorImgViewModel,
    )

    for vm in (IntensityViewModel(), NBViewModel(), MicroTimeViewModel(), PhasorImgViewModel()):
        assert vm.view_spec().sections
        qtbot.addWidget(AutoForm(vm))  # must not raise; qtbot cleans it up


def test_imaging_tools_have_toolbar_and_accept_drops(qtbot):
    """Each imaging tool exposes an action toolbar and accepts file drops."""
    from chisurf.plugins.microscopy.img_pixel_intensity.gui.tool import (
        ImgPixelIntensityTool,
    )
    from chisurf.plugins.microscopy.img_pixel_micro_time.gui.tool import (
        ImgPixelMicroTimeTool,
    )
    from chisurf.plugins.microscopy.img_pixel_nb.gui.tool import ImgPixelNBTool
    from chisurf.plugins.microscopy.img_pixel_phasor.gui.tool import ImgPixelPhasorTool

    for cls in (ImgPixelIntensityTool, ImgPixelNBTool, ImgPixelMicroTimeTool, ImgPixelPhasorTool):
        tool = cls(embedded=True)
        qtbot.addWidget(tool)  # qtbot owns lifecycle -> no teardown segfault
        assert tool.acceptDrops()
        assert len(tool.toolbar.actions()) >= 3  # Run, HDF5, ndxplorer


def test_image_widget_channel_selector_and_movie(qtbot):
    """The general image dock exposes a detector-channel combo + frame-movie controls."""
    from chisurf.gui.autoform.sections.builtin import ImageMapWidget

    class _M:
        def __init__(self):
            self.colormap = "magma"
            self.display_window = "green"
            self.img = None
            self.calls = []

        def window_names(self):
            return ["green", "red"]

        def refresh_display(self, value=None):
            self.calls.append(value)

        def the_map(self):
            return self.img

    m = _M()
    w = ImageMapWidget(
        m, "the_map", colormap=True, colormap_attr="colormap", movie=True,
        channel_source="window_names", channel_attr="display_window",
        channel_call="refresh_display",
    )
    qtbot.addWidget(w)
    assert [w._channel_combo.itemText(i) for i in range(w._channel_combo.count())] == ["green", "red"]

    m.img = np.random.rand(8, 8)  # 2-D → movie disabled
    w.refresh()
    assert not w._play_btn.isEnabled()

    m.img = np.random.rand(4, 8, 8)  # 3-D stack → movie enabled + toggles
    w.refresh()
    assert w._play_btn.isEnabled()
    w._toggle_play()
    assert w._playing
    w._toggle_play()
    assert not w._playing

    w._channel_combo.setCurrentText("red")  # pick writes attr + calls back
    assert m.display_window == "red" and m.calls[-1] == "red"


def test_image_widget_match_2d_orientation(qtbot):
    """``match_2d`` renders a 3D stack frame identically to the 2D map (no transpose)."""
    from chisurf.gui.autoform.sections.builtin import ImageMapWidget

    class _M:
        def __init__(self, img):
            self.img = img

        def the(self):
            return self.img

    def base():
        a = np.zeros((6, 14))  # asymmetric so a transpose is detectable
        a[0, :] = 1.0
        return a

    stored = lambda w: np.asarray(w._image.getImageItem().image)  # noqa: E731

    ref = ImageMapWidget(_M(base()), "the")
    qtbot.addWidget(ref)
    ref.refresh()
    off = ImageMapWidget(_M(base()[None]), "the", movie=True, match_2d=False)
    qtbot.addWidget(off)
    off.refresh()
    on = ImageMapWidget(_M(base()[None]), "the", movie=True, match_2d=True)
    qtbot.addWidget(on)
    on.refresh()

    assert stored(off).shape != stored(ref).shape  # default 3D path transposes
    assert stored(on).shape == stored(ref).shape and np.allclose(stored(on), stored(ref))


@_needs_data
def test_phasor_frames_movie_is_a_gs_stack():
    """The phasor movie source is unstacked per-frame ``g``/``s`` (n,y,x) stacks."""
    from chisurf.plugins.microscopy.img_pixel_phasor.gui.view_model import (
        PhasorImgViewModel,
    )

    vm = PhasorImgViewModel()
    vm.detectors = {"green": {"chs": [0], "ch_p": [], "ch_s": [], "micro_time_ranges": []}}
    vm.display_window = "green"
    vm.filename = _HT3
    assert vm.g_frames() is None  # gated until compute (no UI-thread build pre-Run)
    vm.compute()  # warms the movie caches on the (here inline) compute thread
    g, s = vm.g_frames(), vm.s_frames()
    assert g is not None and s is not None
    assert g.ndim == 3 and g.shape[0] > 1 and g.shape == s.shape
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(s))
    # the phasor-plot movie source: one density per frame, same frame count.
    dens = vm.phasor_histogram_frames(bins=64)
    assert dens is not None and dens.shape == (g.shape[0], 64, 64)
    assert float(dens.max()) > 0.0


def test_phasor_section_movie_controls(qtbot):
    """The phasor section shows frame controls + slider when ``movie`` is set."""
    from chisurf.gui.autoform.sections.phasor_section import PhasorSectionWidget

    class _M:
        PHASOR_G_RANGE = (-0.1, 1.1)
        PHASOR_S_RANGE = (-0.05, 0.7)

        def dens_frames(self):
            return np.random.rand(5, 32, 32)

    w = PhasorSectionWidget(_M(), target=None, movie=True, frames="dens_frames")
    qtbot.addWidget(w)
    w.refresh()
    assert w._movie and w._frames is not None and w._frames.shape[0] == 5
    assert w._slider.maximum() == 4 and w._play_btn.isEnabled()
    w._advance_frame()
    assert w._frame == 1


@_needs_data
def test_mean_micro_time_frames_movie_is_a_stack():
    """The mean-micro-time movie source is an unstacked per-frame (n,y,x) ns stack."""
    from chisurf.plugins.microscopy.img_pixel_micro_time.gui.view_model import (
        MicroTimeViewModel,
    )

    vm = MicroTimeViewModel()
    vm.detectors = {"green": {"chs": [0], "ch_p": [], "ch_s": [], "micro_time_ranges": []}}
    vm.display_window = "green"
    vm.filename = _HT3
    assert vm.mean_micro_time_frames() is None  # gated until compute
    vm.compute()  # warms the movie cache on the (here inline) compute thread
    frames = vm.mean_micro_time_frames()
    assert frames is not None and frames.ndim == 3 and frames.shape[0] > 1
    assert np.all(np.isfinite(frames)) and float(frames.max()) < 1e3


@_needs_data
def test_clsm_draw_opens_imaging_hdf5_via_backref():
    """CLSM Draw resolves an imaging HDF5 back to its source photon data."""
    import tttrlib

    from chisurf.core.fluorescence.imaging import (
        build_clsm,
        intensity_maps,
        maps_to_dataframe,
        write_imaging_hdf5,
    )
    from chisurf.plugins.microscopy.clsm.gui.view_model import ClsmViewModel

    clsm = build_clsm(tttrlib.TTTR(_HT3), channels=[0])
    path = os.path.join(tempfile.gettempdir(), "chisurf_clsm_backref.h5")
    write_imaging_hdf5(maps_to_dataframe(intensity_maps(clsm.get_intensity())), path, source=_HT3)

    vm = ClsmViewModel()
    vm.load_file(path)  # must resolve the .h5 to the source TTTR
    assert vm.filename == _HT3


@_needs_data
def test_nb_per_window_uses_setup_and_mfd_names():
    """N&B computes every setup window; intensity columns use ndxplorer MFD names."""
    from chisurf.plugins.microscopy.img_pixel_nb.gui.view_model import NBViewModel

    vm = NBViewModel()
    vm.apply_setup_settings({"detectors": {
        "green": {"chs": [0, 3], "micro_time_ranges": [(0, 2048)]},
        "red": {"chs": [1, 2], "micro_time_ranges": [(2048, 4095)]},
    }})
    assert vm.window_names() == ["green", "red"]
    vm.filename = _HT3
    vm.run()
    cols = set(vm._columns)
    # N&B writes only its per-window outputs; intensity/count columns are owned
    # by the Intensity tool (so N&B never clobbers them):
    assert "N (green)" in cols and "B (red)" in cols and "epsilon (green)" in cols
    assert not any("kHz" in c or c.startswith("N") and c.endswith("-all") for c in cols)


@_needs_data
def test_intensity_reproduces_columns_removed_from_mle():
    """Every intensity column the MLE stopped emitting is produced by Intensity."""
    from chisurf.plugins.microscopy.img_pixel_intensity.gui.view_model import (
        IntensityViewModel,
    )

    vm = IntensityViewModel()
    vm.apply_setup_settings({"detectors": {
        "green": {"chs": [0, 1], "ch_p": [0], "ch_s": [1], "micro_time_ranges": []},
    }})
    vm.filename = _HT3
    vm.run()
    cols = set(vm._columns)
    # the exact per-detector columns the pixel-MLE used to write:
    removed = {"green Count Rate (KHz)", "Number of Photons", "Ng-p-all", "Ng-s-all", "Ng-all"}
    assert removed <= cols, f"missing from Intensity: {removed - cols}"
    # plus the ndxplorer-MFD name so equations compute derived values:
    assert "S prompt green (kHz)" in cols
    # p + s = all (polarization-resolved counts are consistent):
    import numpy as np

    assert np.allclose(vm._columns["Ng-all"], vm._columns["Ng-p-all"] + vm._columns["Ng-s-all"])


def test_shift_wrap_is_circular():
    """shift_wrap rolls with wrap-around (no counts lost), like the microtime shifter."""
    import numpy as np

    from chisurf.core.fluorescence.imaging import shift_wrap

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(shift_wrap(a, 2), [4.0, 5.0, 1.0, 2.0, 3.0])  # tail wraps to front
    assert abs(shift_wrap(a, 2).sum() - a.sum()) < 1e-9  # circular → total preserved
    assert abs(shift_wrap(a, -1).sum() - a.sum()) < 1e-9


@_needs_data
def test_prepare_irf_splits_vv_vh_and_normalizes_to_unity():
    """IRF prep splits VV/VH, background-corrects, normalizes each to unit area."""
    from chisurf.core.fluorescence.imaging import prepare_irf

    prepared = prepare_irf([_HT3], ch_p=[0], ch_s=[1], n_channels=4096)
    vv, vh = prepared["vv"], prepared["vh"]
    assert vv.min() >= 0.0 and vh.min() >= 0.0  # background-corrected (non-negative)
    assert abs(vv.sum() - 1.0) < 1e-6  # VV normalized to unity
    assert abs(vh.sum() - 1.0) < 1e-6  # VH normalized to unity (independently)


@_needs_data
def test_calibration_view_model_decay_and_conv_range():
    """The IRF & BG view-model builds a decay for the selected detector + conv range."""
    from chisurf.plugins.microscopy.img_calibration.gui.view_model import (
        CalibrationViewModel,
    )

    vm = CalibrationViewModel()
    vm.apply_setup_settings({"detectors": {"green": {"chs": [0]}, "red": {"chs": [1]}}})
    vm.apply_pipeline_context({"source": _HT3})
    assert vm.window_names() == ["green", "red"] and vm.display_detector == "green"
    assert vm.needs_histograms()  # not binned yet (bg thread does it in the GUI)
    vm.ensure_histograms()  # bin (blocking) for the headless test
    decay = vm.decay_data()
    assert decay is not None and len(decay["data"]) > 0
    # drag-set the conv range + per-polarization BG, then publish
    vm.set_conv_range(100, 2000)
    vm.sel_bg_vv = 3.0
    vm.sel_bg_vh = 1.5
    published = {}
    vm.publish = lambda cal: published.update(cal)
    vm.apply()
    assert published["green"]["conv_start"] == 100
    assert published["green"]["conv_stop"] == 2000
    assert published["green"]["bg_vv"] == 3.0 and published["green"]["bg_vh"] == 1.5


@_needs_data
def test_calibration_bg_subtracts_and_changes_signature():
    """The IRF & BG step feeds per-detector background into the intensity compute."""
    import numpy as np

    from chisurf.plugins.microscopy.img_pixel_intensity.gui.view_model import (
        IntensityViewModel,
    )

    det = {"detectors": {"green": {"chs": [0], "ch_p": [0], "ch_s": [], "micro_time_ranges": []}}}
    vm = IntensityViewModel()
    vm.apply_setup_settings(det)
    vm.filename = _HT3
    vm.run()
    before = float(np.nanmean(vm._columns["green Count Rate (KHz)"]))
    sig = vm._signature()

    # per-polarization backgrounds; the total (VV+VH) feeds the intensity subtraction
    vm.apply_calibration({"green": {"irf": [], "bg_vv": before, "bg_vh": 1.0}})
    assert vm._signature() != sig and vm.needs_recompute()
    vm.run()
    after = float(np.nanmean(vm._columns["green Count Rate (KHz)"]))
    assert after < before  # background subtracted (clipped at 0)


@_needs_data
def test_recompute_only_on_change_with_progress():
    """run() reports progress (ending at 1.0) and skips recompute when unchanged."""
    from chisurf.plugins.microscopy.img_pixel_nb.gui.view_model import NBViewModel

    vm = NBViewModel()
    vm.filename = _HT3
    calls = []
    vm.run(progress=lambda f, t: calls.append(f))
    assert vm._columns and calls and calls[-1] == 1.0
    # Inputs unchanged → no recompute, no progress ticks.
    assert not vm.needs_recompute()
    calls2 = []
    vm.run(progress=lambda f, t: calls2.append(f))
    assert calls2 == []


@_needs_data
def test_nb_and_phasor_tools_run():
    """The N&B and phasor tools compute finite maps from a real file."""
    from chisurf.plugins.microscopy.img_pixel_nb.gui.view_model import NBViewModel
    from chisurf.plugins.microscopy.img_pixel_phasor.gui.view_model import (
        PhasorImgViewModel,
    )

    nb = NBViewModel()
    nb.filename = _HT3
    nb.run()  # no setup -> single "ch0" fallback window
    assert nb.n_map() is not None and np.all(np.isfinite(nb.b_map()))
    # per-window columns written to the standard HDF5 (MFD-named intensity)
    assert any(c.startswith("N (") for c in nb._columns)

    ph = PhasorImgViewModel()
    ph.filename, ph.n_ph_min = _HT3, 3
    ph.run()
    assert ph.g_map() is not None and ph.s_map() is not None
    hist = ph.phasor_histogram_map()  # 2-D (g, s) density
    assert hist is not None and hist.ndim == 2
