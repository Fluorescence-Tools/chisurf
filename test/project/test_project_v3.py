import json
import os
import tempfile
import unittest

from chisurf.core.project import Project, save_project, load_project


class TestProjectFormat(unittest.TestCase):

    def test_v4_format_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_v4_project")

            p = Project(
                name="v4_test_project",
                description="Project v4 format test",
                chisurf_version="test-version",
                project_format_version=4,
            )
            p.datasets["ds1"] = {"path": "data/file1.dat", "checksum": "abc123", "uid": "ds-uid-1"}
            p.datasets["ds2"] = {"path": "data/file2.dat", "checksum": "def456", "uid": "ds-uid-2"}
            p.experiments["exp1"] = {"type": "tcspc", "dataset_id": "ds-uid-1", "uid": "exp-uid-1"}
            p.fits.append({
                "uid": "fit-uid-1",
                "name": "fit1",
                "dataset_uid": "ds-uid-1",
                "created": "2024-01-01T00:00:00",
                "local_fits": [
                    {"uid": "lf-uid-1", "name": "local1", "parameters": []}
                ]
            })
            p.ui_state["current_experiment_id"] = "exp-uid-1"
            p.metadata["checkpoint_interval"] = 50

            archive_path = save_project(p, project_dir)
            assert archive_path.is_file()

            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zf:
                raw = json.loads(zf.read("project.json"))

            self.assertEqual(raw["project_format_version"], 4)
            self.assertIn("meta", raw)
            self.assertEqual(raw["meta"]["name"], "v4_test_project")
            self.assertIn("ds1", raw["datasets"])

            loaded = load_project(project_dir)
            self.assertEqual(loaded.project_format_version, 4)
            self.assertEqual(loaded.name, "v4_test_project")
            self.assertEqual(len(loaded.datasets), 2)
            self.assertEqual(len(loaded.fits), 1)
            self.assertEqual(loaded.fits[0]["uid"], "fit-uid-1")

    def test_v4_deterministic_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_ordering")

            p = Project(
                name="ordering_test",
                description="Test",
                project_format_version=4,
            )
            p.datasets["z_dataset"] = {"uid": "z"}
            p.datasets["a_dataset"] = {"uid": "a"}
            p.datasets["m_dataset"] = {"uid": "m"}

            archive_path = save_project(p, project_dir)

            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zf:
                raw = json.loads(zf.read("project.json"))

            dataset_keys = list(raw["datasets"].keys())
            self.assertEqual(dataset_keys, sorted(dataset_keys))

    def test_v4_always_outputs_v4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_v1")

            p = Project(
                name="v4_project",
                description="V4 format",
                project_format_version=4,
            )
            p.datasets["ds1"] = {"path": "data/file1.dat"}

            archive_path = save_project(p, project_dir)

            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zf:
                raw = json.loads(zf.read("project.json"))

            self.assertEqual(raw["project_format_version"], 4)
            self.assertIn("meta", raw)

            loaded = load_project(project_dir)
            self.assertEqual(loaded.project_format_version, 4)
            self.assertEqual(loaded.name, "v4_project")

    def test_get_dataset_by_uid(self):
        p = Project(name="test", project_format_version=4)
        p.datasets["uid1"] = {"name": "dataset1"}
        p.datasets["uid2"] = {"name": "dataset2"}

        self.assertEqual(p.get_dataset("uid1")["name"], "dataset1")
        self.assertIsNone(p.get_dataset("nonexistent"))

    def test_get_fit_by_uid(self):
        p = Project(name="test", project_format_version=4)
        p.fits.append({"uid": "fit1", "name": "Fit 1"})
        p.fits.append({"uid": "fit2", "name": "Fit 2"})

        self.assertEqual(p.get_fit("fit1")["name"], "Fit 1")
        self.assertIsNone(p.get_fit("nonexistent"))

    def test_list_uids(self):
        p = Project(name="test", project_format_version=4)
        p.datasets["z"] = {"uid": "z"}
        p.datasets["a"] = {"uid": "a"}
        p.datasets["m"] = {"uid": "m"}
        p.fits.append({"uid": "fit3"})
        p.fits.append({"uid": "fit1"})
        p.fits.append({"uid": "fit2"})

        self.assertEqual(p.list_dataset_uids(), ["a", "m", "z"])
        self.assertEqual(p.list_fit_uids(), ["fit1", "fit2", "fit3"])


if __name__ == "__main__":
    unittest.main()
