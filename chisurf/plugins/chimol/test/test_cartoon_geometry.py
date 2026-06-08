"""Tests for Chimol cartoon geometry generation.

These tests validate the mesh output from the cartoon extrude pipeline
without requiring Qt or OpenGL.
"""

import numpy as np
import pytest

from chisurf.plugins.chimol.chimol.geometry.cartoon import (
    _generate_cartoon_tube_arrays,
    _sample_path,
    _propagate_ups,
    _build_frames,
    _make_circle_shape,
    _make_oval_shape,
    _make_rectangle_shape,
    _extrude_shape,
    _extrude_arrowhead,
    _segment_ss,
)


# -- Fixtures --

@pytest.fixture
def coords4():
    return np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)


@pytest.fixture
def colors4():
    return np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]],
                    dtype=float)


@pytest.fixture
def coords10():
    return np.array([[i, 0, 0] for i in range(10)], dtype=float)


@pytest.fixture
def colors10():
    return np.tile([1, 0, 0, 1], (10, 1)).astype(float)


# -- Shape profile tests --

class TestShapeProfiles:

    def test_circle_has_expected_verts(self):
        sv, sn = _make_circle_shape(16, 1.0)
        assert sv.shape == (16, 3)
        assert sn.shape == (16, 3)
        # All verts have z=0
        assert np.allclose(sv[:, 0], 0.0)
        # Radius is 1.0
        radii = np.linalg.norm(sv[:, 1:], axis=1)
        assert np.allclose(radii, 1.0)

    def test_circle_normals_are_unit(self):
        _, sn = _make_circle_shape(16, 1.0)
        norms = np.linalg.norm(sn, axis=1)
        assert np.allclose(norms, 1.0)

    def test_oval_shape(self):
        sv, sn = _make_oval_shape(16, 2.0, 3.0)
        assert sv.shape == (16, 3)
        assert sn.shape == (16, 3)
        assert np.allclose(sv[:, 0], 0.0)
        # Check extents
        assert np.max(sv[:, 1]) == pytest.approx(2.0, abs=0.01)
        assert np.max(sv[:, 2]) == pytest.approx(3.0, abs=0.01)

    def test_oval_normals_are_unit(self):
        _, sn = _make_oval_shape(16, 2.0, 3.0)
        norms = np.linalg.norm(sn, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_rectangle_shape(self):
        sv, sn = _make_rectangle_shape(2.0, 0.3)
        assert sv.shape == (8, 3)
        assert sn.shape == (8, 3)
        assert np.allclose(sv[:, 0], 0.0)

    def test_rectangle_normals_are_unit(self):
        _, sn = _make_rectangle_shape(2.0, 0.3)
        norms = np.linalg.norm(sn, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


# -- SS segmenter tests --

class TestSegmenter:

    def test_none_ss_returns_empty(self):
        assert _segment_ss(None) == []
        assert _segment_ss(np.array([])) == []

    def test_all_helix(self):
        segs = _segment_ss(np.array(['H', 'H', 'H'], dtype='U1'))
        assert len(segs) == 1
        assert segs[0]['ss_type'] == 'H'
        assert segs[0] == {'start': 0, 'end': 3, 'ss_type': 'H'}

    def test_mixed(self):
        segs = _segment_ss(np.array(['H', 'H', 'E', 'E', 'C'], dtype='U1'))
        assert len(segs) == 3
        assert segs[0] == {'start': 0, 'end': 2, 'ss_type': 'H'}
        assert segs[1] == {'start': 2, 'end': 4, 'ss_type': 'E'}
        assert segs[2] == {'start': 4, 'end': 5, 'ss_type': 'C'}

    def test_single_residue_types(self):
        segs = _segment_ss(np.array(['H', 'E', 'C'], dtype='U1'))
        assert len(segs) == 3


# -- Extrude tests --

class TestExtrude:

    def test_extrude_circle(self, coords4, colors4):
        path, pc = _sample_path(coords4, colors4, subdivisions=6)
        t, up = _propagate_ups(path, None)
        frames = _build_frames(t, up)
        sv, sn = _make_circle_shape(16, 1.0)
        result = _extrude_shape(path, frames, sv, sn, pc, cap_ends=True)
        assert result is not None
        verts, norms, faces, cols = result
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0
        assert cols is not None
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)
        assert np.all(np.isfinite(verts))

    def test_extrude_oval(self, coords4, colors4):
        path, pc = _sample_path(coords4, colors4, subdivisions=6)
        t, up = _propagate_ups(path, None)
        frames = _build_frames(t, up)
        sv, sn = _make_oval_shape(16, 1.0, 1.5)
        result = _extrude_shape(path, frames, sv, sn, pc, cap_ends=True)
        assert result is not None
        verts, norms, faces, cols = result
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)

    def test_extrude_rectangle(self, coords4, colors4):
        path, pc = _sample_path(coords4, colors4, subdivisions=6)
        t, up = _propagate_ups(path, None)
        frames = _build_frames(t, up)
        sv, sn = _make_rectangle_shape(2.0, 0.3)
        result = _extrude_shape(path, frames, sv, sn, pc, cap_ends=True)
        assert result is not None
        verts, norms, faces, cols = result
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)

    def test_extrude_arrowhead(self, coords4, colors4):
        path, pc = _sample_path(coords4, colors4, subdivisions=6)
        t, up = _propagate_ups(path, None)
        frames = _build_frames(t, up)
        sv, sn = _make_rectangle_shape(2.0, 0.3)
        result = _extrude_arrowhead(path, frames, sv, sn, pc, 2)
        assert result is not None
        verts, norms, faces, cols = result
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)
        assert np.all(np.isfinite(verts))

    def test_short_path_returns_none(self, colors4):
        path = np.zeros((1, 3), dtype=float)
        sv = np.zeros((4, 3), dtype=float)
        sn = np.zeros((4, 3), dtype=float)
        assert _extrude_shape(path, np.zeros((1, 3, 3)), sv, sn, None) is None


# -- Public entry-point tests --

class TestGenerateCartoon:

    def test_tube_style(self, coords4, colors4):
        result = _generate_cartoon_tube_arrays(coords4, colors4, style='tube')
        assert result is not None
        verts, norms, faces, cols = result
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0
        assert cols is not None
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)
        assert np.all(np.isfinite(verts))

    def test_ribbon_no_ss_fallback(self, coords4, colors4):
        result = _generate_cartoon_tube_arrays(coords4, colors4, style='ribbon')
        assert result is not None

    @pytest.mark.parametrize('ss,label', [
        (['H', 'H', 'H', 'H'], 'helix'),
        (['E', 'E', 'E', 'E'], 'strand'),
        (['C', 'C', 'C', 'C'], 'loop'),
    ])
    def test_ribbon_uniform_ss(self, coords4, colors4, ss, label):
        result = _generate_cartoon_tube_arrays(
            coords4, colors4,
            ss_codes=np.array(ss, dtype='U1'),
            style='ribbon',
        )
        assert result is not None, f'{label} returned None'
        verts, norms, faces, cols = result
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)

    def test_ribbon_big_chain(self, coords10, colors10):
        ss = np.array(['C', 'C', 'H', 'H', 'H',
                       'E', 'E', 'E', 'C', 'C'], dtype='U1')
        result = _generate_cartoon_tube_arrays(
            coords10, colors10,
            ss_codes=ss, style='ribbon',
        )
        assert result is not None
        verts, norms, faces, cols = result
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)

    def test_short_chain_returns_none(self, colors4):
        coords = np.zeros((1, 3), dtype=float)
        result = _generate_cartoon_tube_arrays(coords, colors4[:1])
        assert result is None

    def test_no_nans(self, coords10, colors10):
        ss = np.array(['H', 'E', 'C'] * 3 + ['H'], dtype='U1')
        result = _generate_cartoon_tube_arrays(
            coords10, colors10,
            ss_codes=ss, style='ribbon',
        )
        if result is not None:
            verts, norms, faces, cols = result
            assert np.all(np.isfinite(verts))
            assert np.all(np.isfinite(norms))
            assert np.all(np.isfinite(faces))

    def test_single_residue_ss_blocks_do_not_drop_geometry(self, coords4, colors4):
        ss = np.array(['C', 'H', 'E', 'C'], dtype='U1')
        result = _generate_cartoon_tube_arrays(
            coords4, colors4,
            ss_codes=ss, style='ribbon',
        )
        assert result is not None
        verts, norms, faces, cols = result
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0
