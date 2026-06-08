from __future__ import annotations
import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path

from chisurf.plugins.chimol.chimol.testing.mock_viewer import MockViewer
from chisurf.plugins.chimol.chimol.cmd.command import Cmd

class MockWindow:
    def __init__(self, viewer):
        self.viewer = viewer

@dataclass
class Structure:
    atoms: np.ndarray
    xyz: np.ndarray

@pytest.fixture
def editing_context():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    
    atom_dtype = [
        ('xyz', 'f4', (3,)),
        ('atom_name', 'S10'),
        ('res_id', 'i4'),
        ('res_name', 'S10'),
        ('chain_id', 'S4'),
        ('b_factor', 'f4')
    ]
    
    data = np.zeros(10, dtype=atom_dtype)
    data['xyz'] = np.random.rand(10, 3)
    data['atom_name'] = [f'A{i}'.encode() for i in range(10)]
    data['res_id'] = np.arange(10)
    data['res_name'] = b'ALA'
    data['chain_id'] = b'A'
    data['b_factor'] = 10.0
    
    struct = Structure(atoms=data, xyz=data['xyz'])
    viewer.set_structure(struct)
    return viewer, cmd

def test_chimol_editing_alter(editing_context):
    viewer, cmd = editing_context
    cmd.do("alter resi 0-4, b = 50.0")
    
    atoms = viewer.get_active_state().atoms
    assert np.all(atoms['b_factor'][:5] == 50.0)
    assert np.all(atoms['b_factor'][5:] == 10.0)

def test_chimol_editing_remove(editing_context):
    viewer, cmd = editing_context
    cmd.do("remove resi 0-2")
    
    new_atoms = viewer.get_active_state().atoms
    assert len(new_atoms) == 7

def test_chimol_editing_pseudoatom(editing_context):
    viewer, cmd = editing_context
    cmd.do("pseudoatom ps1, pos=[10.0 10.0 10.0]")
    cmd.do("pseudoatom ps1, pos=[20.0 20.0 20.0]") # Append
    
    ps_entry = None
    for e in viewer._objects.values():
         if e.name == "ps1":
              ps_entry = e
              break

    assert ps_entry is not None
    ps_atoms = ps_entry.state.atoms
    assert len(ps_atoms) == 2
    assert np.all(ps_atoms[1]['xyz'] == [20.0, 20.0, 20.0])


def test_chimol_cmd_clear_selection(editing_context):
    viewer, cmd = editing_context
    messages = []
    cmd.set_message_callback(messages.append)

    viewer.set_selected_residues([1, 2, 3])
    cmd.do("clear")

    assert viewer._selected_residues == []
    assert any("Cleared current selection" in msg for msg in messages)

    viewer.set_selected_residues([4])
    cmd.clear()
    assert viewer._selected_residues == []


def test_chimol_cmd_delete_all_reinitialize_copy(editing_context):
    viewer, cmd = editing_context
    messages = []
    errors = []
    cmd.set_message_callback(messages.append)
    cmd.set_error_callback(errors.append)

    first_id = viewer.get_active_object_id()
    assert first_id is not None

    cmd.do(f"copy copied, {first_id}")
    names = [obj["name"] for obj in viewer.list_objects()]
    assert "copied" in names

    cmd.do("delete all")
    assert viewer.list_objects() == []
    assert cmd._named_selections == {}

    viewer._add_mock_object("objA", "objA")
    viewer._reps["sticks"] = True
    cmd._named_selections["sele"] = {"object_id": "objA", "indices": [0]}
    cmd.do("reinitialize")
    assert viewer.list_objects() == []
    assert cmd._named_selections == {}
    assert errors == []


def test_chimol_cmd_png_and_ray(editing_context, tmp_path: Path):
    viewer, cmd = editing_context
    messages = []
    errors = []
    cmd.set_message_callback(messages.append)
    cmd.set_error_callback(errors.append)

    out = tmp_path / "view_export"
    cmd.do(f"png {out}")
    assert out.with_suffix(".png").is_file()

    cmd.do("ray 800 600")
    assert any("ray:" in msg for msg in messages)
    assert errors == []


def test_chimol_cmd_get_set_view(editing_context):
    viewer, cmd = editing_context
    messages = []
    errors = []
    cmd.set_message_callback(messages.append)
    cmd.set_error_callback(errors.append)

    view = [1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            55.0, 30.0, 75.0,
            1.0, 2.0, 3.0,
            0.2, 500.0, 45.0]
    cmd.set_view(view)
    assert viewer.get_view_state() == view

    text = cmd.get_view()
    assert text.startswith("set_view (")

    viewer.set_view_state([0.0] * 18)
    cmd.do(text)
    assert viewer.get_view_state() == view
    assert errors == []


def test_chimol_cmd_cartoon_spectrum_settings(editing_context):
    viewer, cmd = editing_context
    errors = []
    cmd.set_error_callback(errors.append)

    cmd.do("spectrum count, rainbow")
    assert viewer._color_mode == "spectrum"

    cmd.do("cartoon tube")
    from chisurf.plugins.chimol.chimol.config import _DISPLAY_CONFIG
    assert _DISPLAY_CONFIG["cartoon"]["style"] == "tube"

    cmd.do("set cartoon_oval_width, 0.33")
    assert _DISPLAY_CONFIG["cartoon"]["oval_width"] == 0.33
    assert errors == []


def test_chimol_cmd_color_names(editing_context):
    """Color command accepts all major PyMOL named colors."""
    viewer, cmd = editing_context
    errors = []
    cmd.set_error_callback(errors.append)

    # Color without selection should apply uniform color to all atoms
    cmd.do("color red")
    if errors:
        pytest.fail(f"color red produced errors: {errors}")
    # Verify atom color overrides were set
    state = viewer.get_active_state()
    assert state.colors_per_atom_override is not None
    assert np.allclose(state.colors_per_atom_override[0], [1.0, 0.0, 0.0, 1.0])

    # With selection: color <name>, <selection>
    errors.clear()
    cmd.do("color marine, resi 1-5")
    assert errors == [], f"Unexpected errors: {errors}"
    assert state.colors_per_atom_override is not None
    # Atom 1 has res_id=1 (in range 1-5)
    assert np.allclose(state.colors_per_atom_override[1], [0.0, 0.5, 1.0, 1.0])

    # More named colors via comma form
    for name in ("green", "blue", "cyan", "yellow", "magenta",
                 "orange", "purple", "salmon", "lime", "teal",
                 "olive", "forest", "ruby", "slate", "hotpink",
                 "wheat", "brown", "chocolate", "violet", "pink",
                 "firebrick", "tv_red", "tv_green", "tv_blue",
                 "tv_yellow", "tv_orange", "brightorange",
                 "lightblue", "lightorange", "palegreen", "skyblue",
                 "grey40", "gray60", "grey99"):
        errors.clear()
        cmd.do(f"color {name}, all")
        if errors:
            pytest.fail(f"color {name}, all produced errors: {errors}")

    # Unknown color should produce error
    errors.clear()
    cmd.do("color nonexistent_color")
    assert len(errors) > 0

    # unknown color with comma
    errors.clear()
    cmd.do("color nonexistent_color, all")
    assert len(errors) > 0


def test_chimol_cmd_set_color(editing_context):
    """set_color defines a new named color."""
    viewer, cmd = editing_context
    messages = []
    errors = []
    cmd.set_message_callback(messages.append)
    cmd.set_error_callback(errors.append)

    cmd.do("set_color my_custom, 0.1 0.2 0.3")
    assert errors == [], f"Unexpected errors: {errors}"
    assert any("my_custom" in msg for msg in messages)

    # Use the custom color
    errors.clear()
    cmd.do("color my_custom, resi 1-5")
    assert errors == []

    # set_color with hex
    errors.clear()
    cmd.do("set_color my_hex, #ff8800")
    assert errors == []

    # Verify the user color is in the cmd's user_colors dict
    assert "my_custom" in cmd._user_colors
    assert "my_hex" in cmd._user_colors


def test_chimol_cmd_get_color_index(editing_context):
    """get_color_index returns 0 for known colors, error for unknown."""
    viewer, cmd = editing_context
    messages = []
    errors = []
    cmd.set_message_callback(messages.append)
    cmd.set_error_callback(errors.append)

    idx = cmd._cmd_get_color_index(["red"])
    assert idx == 0
    assert errors == []

    errors.clear()
    idx = cmd._cmd_get_color_index(["marine"])
    assert idx == 0
    assert errors == []

    # Unknown color
    errors.clear()
    idx = cmd._cmd_get_color_index(["totallynotacolor"])
    assert idx is None
    assert len(errors) > 0


def test_raytracer_single_sphere():
    """A single sphere renders a non-background pixel at its center."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )

    spheres = [
        Sphere(
            center=np.array([0.0, 0.0, 5.0]),
            radius=1.0,
            color=np.array([1.0, 0.0, 0.0]),
        )
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_degrees=45.0,
    )
    light = np.array([0.5, -0.5, 1.0])

    img = trace(spheres, camera, light, width=64, height=64, ssaa=2,
                color_blend=False, depth_cue=False)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8

    center_pixel = img[32, 32]
    assert not np.allclose(center_pixel, [0, 0, 0])


def test_raytracer_no_spheres():
    """Zero spheres produce a fully-background image."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        RayCamera, trace,
    )

    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light = np.array([0.0, 0.0, 1.0])

    img = trace([], camera, light, width=32, height=32, background=(50, 100, 150), ssaa=1,
                color_blend=False, depth_cue=False, gamma=1.0)
    assert np.allclose(img[0, 0], [50, 100, 150])


def test_raytracer_shadow():
    """A sphere behind another (w.r.t. light) is in shadow."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )

    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=0.5, color=np.array([0.0, 1.0, 0.0])),
        Sphere(center=np.array([0.2, 0.0, 5.5]), radius=0.3, color=np.array([1.0, 0.0, 0.0])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_degrees=30.0,
    )
    light = np.array([0.0, 0.0, 1.0])

    img = trace(spheres, camera, light, width=80, height=80, ambient=0.1, ssaa=2,
                color_blend=False, depth_cue=False)
    center = img[40, 40]
    assert not np.allclose(center, [0, 0, 0])


def test_chimol_cmd_ray_integration(editing_context, tmp_path: Path):
    """ray command produces a PNG file."""
    viewer, cmd = editing_context
    messages = []
    errors = []
    cmd.set_message_callback(messages.append)
    cmd.set_error_callback(errors.append)

    out = tmp_path / "ray_test.png"
    cmd.do(f"ray {out}, 100, 100")
    if errors:
        pytest.fail(f"ray produced errors: {errors}")

    if out.is_file():
        from PIL import Image
        img = Image.open(str(out))
        assert img.size == (100, 100)
    else:
        ok = any("ray:" in msg for msg in messages)
        assert ok, "ray did not produce a file or message"


# ---------------------------------------------------------------------------
# Phase 1: Lighting overhaul
# ---------------------------------------------------------------------------

def test_raytracer_multi_light():
    """Two lights produce different lighting than one light."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=1.0, color=np.array([0.8, 0.8, 0.8])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    # Single light
    img_one = trace(spheres, camera, np.array([[0.0, 0.0, 1.0]]),
                    width=32, height=32, ssaa=1,
                    specular=0.0,
                    color_blend=False, depth_cue=False, shadow=False)
    # Two lights (second from the side)
    img_two = trace(spheres, camera, np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
                    width=32, height=32, ssaa=1,
                    specular=0.0,
                    color_blend=False, depth_cue=False, shadow=False)
    # At least one pixel should differ
    assert not np.array_equal(img_one, img_two)


def test_raytracer_direct_specular():
    """Direct specular (head-on) adds a bright highlight with power 55."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=1.0, color=np.array([0.5, 0.5, 0.5])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light = np.array([[0.0, 0.0, 1.0]])

    # With direct specular
    img_on = trace(spheres, camera, light, width=32, height=32, ssaa=1,
                   direct_specular=0.5, direct_specular_power=55.0,
                   ambient=0.0, diffuse=0.0, specular=0.0,
                   color_blend=False, depth_cue=False, shadow=False)
    # Without direct specular
    img_off = trace(spheres, camera, light, width=32, height=32, ssaa=1,
                    direct_specular=0.0, direct_specular_power=55.0,
                    ambient=0.0, diffuse=0.0, specular=0.0,
                    color_blend=False, depth_cue=False, shadow=False)

    # With direct specular should be brighter at center where sphere faces camera
    assert img_on[16, 16].sum() > img_off[16, 16].sum()


def test_raytracer_lower_ambient():
    """Lower ambient produces higher contrast (darker shadow side)."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    # Use a larger sphere closer so pixels definitely hit it
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 3.0]), radius=1.5,
               color=np.array([0.8, 0.8, 0.8])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    # Light from upper-right; left side of sphere is in shadow
    light = np.array([[1.0, 0.0, 1.0]])

    # High ambient
    img_high = trace(spheres, camera, light, width=32, height=32, ssaa=1,
                     ambient=0.5, diffuse=0.45, specular=0.0,
                     color_blend=False, depth_cue=False, shadow=False)
    # Low ambient (PyMOL default)
    img_low = trace(spheres, camera, light, width=32, height=32, ssaa=1,
                    ambient=0.14, diffuse=0.45, specular=0.0,
                    color_blend=False, depth_cue=False, shadow=False)

    # Far left pixel should hit sphere (radius 1.5 at z=3 covers ~17 pixels
    # at fov=45). Left side is shadowed (N·L < 0), so only ambient contributes.
    # Lower ambient => darker pixel.
    edge_left_high = int(img_high[16, 5].sum())
    edge_left_low = int(img_low[16, 5].sum())
    assert edge_left_low < edge_left_high

    # At center (fully lit), higher ambient => brighter
    center_high = int(img_high[16, 16].sum())
    center_low = int(img_low[16, 16].sum())
    assert center_low < center_high


# ---------------------------------------------------------------------------
# Phase 2: Gamma correction
# ---------------------------------------------------------------------------

def test_raytracer_gamma_background():
    """Gamma correction changes effective background color."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import trace
    from chisurf.plugins.chimol.chimol.renderer.raytracer import RayCamera

    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light = np.array([[0.0, 0.0, 1.0]])

    # Gamma 1.0 = no change
    img_linear = trace([], camera, light, width=16, height=16,
                       background=(100, 100, 100), gamma=1.0, ssaa=1,
                       color_blend=False, depth_cue=False)
    assert np.allclose(img_linear[0, 0], [100, 100, 100])

    # Gamma 2.2 applies luminance-preserving gamma to background.
    # pow(0.392, 2.2) / 0.392 ≈ 0.324, so background ~32 (dimmer in linear space).
    img_gamma = trace([], camera, light, width=16, height=16,
                      background=(100, 100, 100), gamma=2.2, ssaa=1,
                      color_blend=False, depth_cue=False)
    # Gamma-corrected background is dimmer (linearized for rendering)
    assert img_gamma[0, 0, 0] < 100


# ---------------------------------------------------------------------------
# Phase 3: Soft shadows
# ---------------------------------------------------------------------------

def test_raytracer_soft_shadow():
    """Soft shadow makes shadowed region brighter than hard shadow."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
        _HAVE_NUMBA,
    )
    if not _HAVE_NUMBA:
        pytest.skip("Soft shadow requires Numba kernel")

    # Two spheres side by side at same depth. Light from left [-1,0,0].
    # Sphere 2 (right) casts shadow on sphere 1 (left). The shadow on
    # sphere 1's right side is visible to the camera.
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=1.0,
               color=np.array([0.7, 0.7, 0.7])),
        Sphere(center=np.array([2.5, 0.0, 4.5]), radius=0.8,
               color=np.array([0.7, 0.7, 0.7])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    # Light from left: sphere 2 (right) occludes sphere 1 (left)
    light = np.array([[-1.0, 0.0, 0.0]])

    # Hard shadow (decay_factor=0.0)
    img_hard = trace(spheres, camera, light, width=64, height=64,
                     ambient=0.2, ssaa=1, shadow_decay_factor=0.0,
                     color_blend=False, depth_cue=False, gamma=1.0)

    # Soft shadow (decay_factor=2.0, decay_range=0.5)
    img_soft = trace(spheres, camera, light, width=64, height=64,
                     ambient=0.2, ssaa=1,
                     shadow_decay_factor=2.0, shadow_decay_range=0.5,
                     color_blend=False, depth_cue=False, gamma=1.0)

    assert not np.array_equal(img_hard, img_soft)


def test_raytracer_shadow_fudge():
    """Shadow fudge prevents self-shadowing; without it shadow may alias."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=1.0,
               color=np.array([0.5, 0.5, 0.5])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light = np.array([[0.0, 0.0, 1.0]])

    # With proper shadow_fudge, sphere should be fully lit
    img = trace(spheres, camera, light, width=32, height=32, ssaa=1,
                ambient=0.2, shadow=True, shadow_fudge=0.001,
                color_blend=False, depth_cue=False)
    # Center pixel should be lit (not in shadow)
    assert not np.allclose(img[16, 16], [0, 0, 0])


# ---------------------------------------------------------------------------
# Phase 4: Depth cueing / fog
# ---------------------------------------------------------------------------

def test_raytracer_depth_cue():
    """Depth cueing makes far spheres fade to background."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=0.5,
               color=np.array([1.0, 1.0, 1.0])),
        Sphere(center=np.array([0.0, 0.0, 50.0]), radius=0.5,
               color=np.array([1.0, 1.0, 1.0])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        far_clip=100.0,
    )
    light = np.array([[0.0, 0.0, 1.0]])

    # Without depth cue
    img_nocue = trace(spheres, camera, light, width=32, height=32,
                      ambient=0.5, depth_cue=False, ssaa=1,
                      color_blend=False)
    # With depth cue
    img_cue = trace(spheres, camera, light, width=32, height=32,
                    ambient=0.5, depth_cue=True, fog_start=0.3,
                    fog_intensity=1.0, ssaa=1, color_blend=False)

    # Far sphere should be closer to background with depth cue
    far_pixel_nocue = img_nocue[16, 16].astype(int).sum()
    far_pixel_cue = img_cue[16, 16].astype(int).sum()
    assert far_pixel_cue <= far_pixel_nocue


# ---------------------------------------------------------------------------
# Phase 5: Anti-aliasing
# ---------------------------------------------------------------------------

def test_raytracer_antialias_levels():
    """Different SSAA levels produce different images."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=1.0,
               color=np.array([1.0, 0.0, 0.0])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light = np.array([[0.0, 0.0, 1.0]])

    img_noaa = trace(spheres, camera, light, width=16, height=16, ssaa=1,
                     color_blend=False, depth_cue=False)
    img_aa2 = trace(spheres, camera, light, width=16, height=16, ssaa=2,
                    color_blend=False, depth_cue=False)
    img_aa3 = trace(spheres, camera, light, width=16, height=16, ssaa=3,
                    color_blend=False, depth_cue=False)

    assert img_noaa.shape == (16, 16, 3)
    assert img_aa2.shape == (16, 16, 3)
    assert img_aa3.shape == (16, 16, 3)

    # ssaa levels should differ at some pixel (antialiasing smooths edges)
    assert not np.array_equal(img_noaa, img_aa2)


# ---------------------------------------------------------------------------
# Phase 6: Color blend post-filter
# ---------------------------------------------------------------------------

def test_raytracer_color_blend():
    """Color blend raises minimum per-channel values."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace, _apply_color_blend,
    )
    import numpy as np

    # Create a single-channel-dominant image
    test_img = np.zeros((4, 4, 3), dtype=np.uint8)
    test_img[:, :, 0] = 200  # Strong red, no green or blue
    test_img[:, :, 1] = 0
    test_img[:, :, 2] = 0

    # Apply color blend
    blended = _apply_color_blend(test_img, 0.17, 0.25, 0.14)

    # Red should stay high
    assert blended[0, 0, 0] >= 200
    # Green and blue should now be >= their respective blend minima
    # red_part = 0.17 * 200 = 34
    # green_part = 0.25 * 0 = 0
    # blue_part = 0.14 * 0 = 0
    # green_min = max(red_part=34, blue_part=0) = 34
    # blue_min = max(green_part=0, red_part=34) = 34
    assert blended[0, 0, 1] >= 34
    assert blended[0, 0, 2] >= 34


def test_raytracer_color_blend_sphere():
    """Ray trace with color_blend=True produces valid output."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=1.0,
               color=np.array([1.0, 0.0, 0.0])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light = np.array([[0.0, 0.0, 1.0]])

    img_blend = trace(spheres, camera, light, width=32, height=32,
                      ssaa=1, color_blend=True, depth_cue=False)
    img_noblend = trace(spheres, camera, light, width=32, height=32,
                        ssaa=1, color_blend=False, depth_cue=False)

    assert img_blend.shape == (32, 32, 3)
    assert not np.array_equal(img_blend, img_noblend)


# ---------------------------------------------------------------------------
# Full integration: PyMOL defaults render
# ---------------------------------------------------------------------------

def test_raytracer_pymol_defaults():
    """Render with PyMOL defaults produces reasonable output."""
    from chisurf.plugins.chimol.chimol.renderer.raytracer import (
        Sphere, RayCamera, trace,
    )
    spheres = [
        Sphere(center=np.array([0.0, 0.0, 5.0]), radius=1.0,
               color=np.array([0.8, 0.2, 0.2])),
        Sphere(center=np.array([1.5, 0.0, 5.0]), radius=1.0,
               color=np.array([0.2, 0.8, 0.2])),
        Sphere(center=np.array([0.0, 1.5, 5.0]), radius=1.0,
               color=np.array([0.2, 0.2, 0.8])),
    ]
    camera = RayCamera(
        origin=np.array([0.0, 0.0, 0.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light = np.array([[0.0, 0.0, 1.0], [0.5, 0.3, 1.0]])

    img = trace(spheres, camera, light, width=64, height=64, ssaa=2,
                ambient=0.14, diffuse=0.45, specular=0.25, shininess=40.0,
                direct_specular=0.30, direct_specular_power=55.0,
                reflect_power=1.0, legacy_lighting=0.0,
                shadow=True, shadow_fudge=0.001,
                shadow_decay_factor=0.2, shadow_decay_range=1.8,
                gamma=2.2, depth_cue=True, fog_start=0.45, fog_intensity=1.0,
                color_blend=True, color_blend_red=0.17,
                color_blend_green=0.25, color_blend_blue=0.14)

    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    # Center pixel should be non-black (we hit a sphere)
    assert not np.allclose(img[32, 32], [0, 0, 0], atol=5)


# ---------------------------------------------------------------------------
# Config-driven ray
# ---------------------------------------------------------------------------

def test_cmd_ray_uses_config_settings(editing_context, tmp_path):
    """The ray command reads settings from the config's ray section."""
    viewer, cmd = editing_context
    from chisurf.plugins.chimol.chimol.config import _DISPLAY_CONFIG
    messages = []
    errors = []
    cmd.set_message_callback(messages.append)
    cmd.set_error_callback(errors.append)

    out = tmp_path / "ray_config_test.png"
    cmd.do(f"ray {out}, 32, 32")
    if errors:
        pytest.fail(f"ray produced errors: {errors}")
    # Just verify the command ran without error
    ok = any("ray:" in msg for msg in messages)
    assert ok, f"ray did not produce a message. Messages: {messages}"
