import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import chisurf.core.structure as cs_struct
from chisurf.plugins.chimol.chimol.geometry.primitives import _compute_center_radius
from chisurf.plugins.chimol.chimol.renderer.raytracer import (
    RayCamera,
    Sphere,
    trace,
)


class TestVisualRendering(unittest.TestCase):
    """Test suite for visual rendering of molecular structures.

    This test suite performs headless ray-tracing on structure coordinates
    to verify output image dimensions and verify that non-blank renders are generated.
    """

    def setUp(self) -> None:
        """Set up directories and resolve coordinate file paths for testing."""
        here = Path(__file__).resolve()
        # Find repo root
        self._repo_root = here.parents[4]

        self._pdb_148l = self._repo_root / "test" / "data" / "atomic_coordinates" / "pdb_files" / "148l.pdb"
        self._pdb_1rtd = self._repo_root / "test" / "data" / "atomic_coordinates" / "pdb_files" / "1rtd.pdb"

        self._renders_dir = self._repo_root / "test_renders"
        self._renders_dir.mkdir(exist_ok=True)

    def _get_spheres_from_structure(self, struct: cs_struct.Structure) -> list[Sphere]:
        """Convert a Structure into a list of Sphere objects colored by element."""
        cpk_colors = {
            'C': [0.5, 0.5, 0.5],
            'O': [1.0, 0.0, 0.0],
            'N': [0.0, 0.0, 1.0],
            'S': [1.0, 1.0, 0.0],
            'P': [1.0, 0.6, 0.0],
            'H': [0.9, 0.9, 0.9],
        }

        positions = struct.xyz
        radii = struct.vdw

        spheres = []
        for i in range(len(positions)):
            elem = struct.atoms[i]['element']
            if isinstance(elem, bytes):
                elem_str = elem.decode('utf-8', errors='ignore').strip().upper()
            else:
                elem_str = str(elem).strip().upper()

            color = cpk_colors.get(elem_str, [1.0, 0.75, 0.8])
            radius = float(radii[i]) if radii is not None and i < len(radii) else 1.0
            if not np.isfinite(radius) or radius <= 0:
                radius = 1.0

            spheres.append(Sphere(
                center=positions[i],
                radius=radius,
                color=np.array(color, dtype=np.float64),
            ))
        return spheres

    def test_render_148l_protein(self) -> None:
        """Render T4 Lysozyme (148L) protein and save the image."""
        self.assertTrue(self._pdb_148l.is_file(), f"Structure not found: {self._pdb_148l}")

        struct = cs_struct.Structure(str(self._pdb_148l))
        self.assertGreater(len(struct.atoms), 0)

        spheres = self._get_spheres_from_structure(struct)
        center, radius = _compute_center_radius(struct.xyz)

        # Position camera looking at the structure
        camera = RayCamera(
            origin=center + np.array([0.0, 0.0, radius * 3.5], dtype=np.float64),
            forward=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            fov_degrees=45.0,
            far_clip=radius * 10.0,
        )

        # Perform ray tracing
        light_dirs = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]], dtype=np.float64)
        img = trace(
            spheres, camera, light_dirs,
            width=256, height=256,
            background=(30, 30, 30),
            ssaa=2,
        )

        # Verify output image properties
        self.assertEqual(img.shape, (256, 256, 3))
        self.assertEqual(img.dtype, np.uint8)

        # Ensure it's not just a blank background image
        bg_count = np.sum((img[:, :, 0] == 30) & (img[:, :, 1] == 30) & (img[:, :, 2] == 30))
        total_pixels = img.shape[0] * img.shape[1]
        self.assertLess(bg_count, total_pixels, "Rendered image is empty/only background.")

        # Save to disk
        out_path = self._renders_dir / "148l_protein.png"
        Image.fromarray(img).save(str(out_path))
        print(f"\nSaved 148L render to: {out_path}")

    def test_render_1rtd_protein_dna(self) -> None:
        """Render HIV Reverse Transcriptase (1RTD) protein + DNA complex and save the image."""
        self.assertTrue(self._pdb_1rtd.is_file(), f"Structure not found: {self._pdb_1rtd}")

        struct = cs_struct.Structure(str(self._pdb_1rtd))
        self.assertGreater(len(struct.atoms), 0)

        spheres = self._get_spheres_from_structure(struct)
        center, radius = _compute_center_radius(struct.xyz)

        # Position camera looking at the structure
        camera = RayCamera(
            origin=center + np.array([0.0, 0.0, radius * 3.5], dtype=np.float64),
            forward=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            fov_degrees=45.0,
            far_clip=radius * 10.0,
        )

        # Perform ray tracing
        light_dirs = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]], dtype=np.float64)
        img = trace(
            spheres, camera, light_dirs,
            width=256, height=256,
            background=(30, 30, 30),
            ssaa=2,
        )

        # Verify output image properties
        self.assertEqual(img.shape, (256, 256, 3))
        self.assertEqual(img.dtype, np.uint8)

        # Ensure it's not just a blank background image
        bg_count = np.sum((img[:, :, 0] == 30) & (img[:, :, 1] == 30) & (img[:, :, 2] == 30))
        total_pixels = img.shape[0] * img.shape[1]
        self.assertLess(bg_count, total_pixels, "Rendered image is empty/only background.")

        # Save to disk
        out_path = self._renders_dir / "1rtd_complex.png"
        Image.fromarray(img).save(str(out_path))
        print(f"\nSaved 1RTD render to: {out_path}")

    def test_render_148l_ses(self) -> None:
        """Render T4 Lysozyme (148L) Solvent Excluded Surface (SES)."""
        from chisurf.plugins.chimol.chimol.geometry.surface import _generate_surface_mesh_edt
        self.assertTrue(self._pdb_148l.is_file(), f"Structure not found: {self._pdb_148l}")

        struct = cs_struct.Structure(str(self._pdb_148l))
        self.assertGreater(len(struct.atoms), 0)

        # Generate SES surface mesh
        mesh_data = _generate_surface_mesh_edt(
            struct.xyz,
            struct.vdw,
            method="ses",
            probe_radius=1.4,
            grid_spacing=1.0,
            padding=3.0,
            max_dim=96,
        )
        self.assertIsNotNone(mesh_data, "SES mesh generation returned None.")
        verts, faces, norms = mesh_data
        self.assertGreater(verts.shape[0], 0)
        self.assertGreater(faces.shape[0], 0)
        self.assertEqual(norms.shape, verts.shape)

        # Represent mesh vertices as spheres for visual rendering
        spheres = [
            Sphere(
                center=verts[i],
                radius=0.25,
                color=np.array([0.7, 0.8, 0.95], dtype=np.float64),
            )
            for i in range(verts.shape[0])
        ]

        center, radius = _compute_center_radius(struct.xyz)

        # Position camera looking at the structure
        camera = RayCamera(
            origin=center + np.array([0.0, 0.0, radius * 3.5], dtype=np.float64),
            forward=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            fov_degrees=45.0,
            far_clip=radius * 10.0,
        )

        # Perform ray tracing
        light_dirs = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]], dtype=np.float64)
        img = trace(
            spheres, camera, light_dirs,
            width=256, height=256,
            background=(30, 30, 30),
            ssaa=2,
        )

        # Verify output image properties
        self.assertEqual(img.shape, (256, 256, 3))
        self.assertEqual(img.dtype, np.uint8)

        # Ensure it's not just a blank background image
        bg_count = np.sum((img[:, :, 0] == 30) & (img[:, :, 1] == 30) & (img[:, :, 2] == 30))
        total_pixels = img.shape[0] * img.shape[1]
        self.assertLess(bg_count, total_pixels, "Rendered SES image is empty/only background.")

        # Save to disk
        out_path = self._renders_dir / "148l_ses.png"
        Image.fromarray(img).save(str(out_path))
        print(f"\nSaved 148L SES surface render to: {out_path}")

    def test_render_1rtd_sas(self) -> None:
        """Render HIV Reverse Transcriptase (1RTD) Solvent Accessible Surface (SAS)."""
        from chisurf.plugins.chimol.chimol.geometry.surface import _generate_surface_mesh_edt
        self.assertTrue(self._pdb_1rtd.is_file(), f"Structure not found: {self._pdb_1rtd}")

        struct = cs_struct.Structure(str(self._pdb_1rtd))
        self.assertGreater(len(struct.atoms), 0)

        # Generate SAS surface mesh
        mesh_data = _generate_surface_mesh_edt(
            struct.xyz,
            struct.vdw,
            method="sas",
            probe_radius=1.4,
            grid_spacing=1.2,
            padding=3.0,
            max_dim=96,
        )
        self.assertIsNotNone(mesh_data, "SAS mesh generation returned None.")
        verts, faces, norms = mesh_data
        self.assertGreater(verts.shape[0], 0)
        self.assertGreater(faces.shape[0], 0)
        self.assertEqual(norms.shape, verts.shape)

        # Represent mesh vertices as spheres for visual rendering
        spheres = [
            Sphere(
                center=verts[i],
                radius=0.3,
                color=np.array([0.9, 0.75, 0.7], dtype=np.float64),
            )
            for i in range(verts.shape[0])
        ]

        center, radius = _compute_center_radius(struct.xyz)

        # Position camera looking at the structure
        camera = RayCamera(
            origin=center + np.array([0.0, 0.0, radius * 3.5], dtype=np.float64),
            forward=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            fov_degrees=45.0,
            far_clip=radius * 10.0,
        )

        # Perform ray tracing
        light_dirs = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]], dtype=np.float64)
        img = trace(
            spheres, camera, light_dirs,
            width=256, height=256,
            background=(30, 30, 30),
            ssaa=2,
        )

        # Verify output image properties
        self.assertEqual(img.shape, (256, 256, 3))
        self.assertEqual(img.dtype, np.uint8)

        # Ensure it's not just a blank background image
        bg_count = np.sum((img[:, :, 0] == 30) & (img[:, :, 1] == 30) & (img[:, :, 2] == 30))
        total_pixels = img.shape[0] * img.shape[1]
        self.assertLess(bg_count, total_pixels, "Rendered SAS image is empty/only background.")

        # Save to disk
        out_path = self._renders_dir / "1rtd_sas.png"
        Image.fromarray(img).save(str(out_path))
        print(f"\nSaved 1RTD SAS surface render to: {out_path}")

    def test_1rtd_trace_extraction(self) -> None:
        """Verify that trace extraction for 1RTD returns both protein and nucleic residues."""
        from chisurf.plugins.chimol.chimol.geometry.trace import _extract_ca_trace
        self.assertTrue(self._pdb_1rtd.is_file(), f"Structure not found: {self._pdb_1rtd}")

        struct = cs_struct.Structure(str(self._pdb_1rtd))
        self.assertGreater(len(struct.atoms), 0)

        # Call the trace extraction helper on the raw atoms
        coords, res_ids, res_names, chain_ids = _extract_ca_trace(struct.atoms)

        self.assertIsNotNone(coords, "Trace extraction returned None.")
        self.assertGreater(coords.shape[0], 0)
        self.assertEqual(len(res_ids), coords.shape[0])
        self.assertEqual(len(chain_ids), coords.shape[0])

        # Get unique chain IDs in the trace
        unique_chains = set(chain_ids)
        print(f"\nUnique chains in 1RTD trace: {unique_chains}")

        # In 1RTD:
        # - Chains A and B are the HIV-1 RT protein subunits.
        # - Chains A and B must be present, along with nucleic acid chains (T, P, etc.).
        self.assertTrue(
            "A" in unique_chains or "B" in unique_chains,
            "Protein chains (A/B) are missing from the extracted trace!"
        )
        self.assertTrue(
            any(ch not in ("A", "B") for ch in unique_chains),
            "DNA chains are missing from the extracted trace!"
        )


if __name__ == "__main__":
    unittest.main()

