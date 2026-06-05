import json
import os
import tempfile
import unittest

from chisurf.core.project import Project, save_project, load_project


class TestProjectV3Format(unittest.TestCase):

    def test_v3_format_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_v3_project")

            p = Project(
                name="v3_test_project",
                description="Project v3 format test",
                chisurf_version="test-version",
                project_format_version=3,
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
            p.links.append({
                "source_fit_uid": "fit-uid-1",
                "source_param_uid": "p-uid-1",
                "target_fit_uid": "fit-uid-2",
                "target_param_uid": "p-uid-2",
            })
            p.ui_state["current_experiment_id"] = "exp-uid-1"
            p.metadata["checkpoint_interval"] = 50

            project_json_path = save_project(p, project_dir)
            assert project_json_path.is_file()

            with project_json_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            self.assertEqual(raw["project_format_version"], 3)
            self.assertIn("meta", raw)
            self.assertEqual(raw["meta"]["name"], "v3_test_project")
            self.assertIn("ds1", raw["datasets"])
            self.assertIn("links", raw)

            loaded = load_project(project_dir)
            self.assertEqual(loaded.project_format_version, 3)
            self.assertEqual(loaded.name, "v3_test_project")
            self.assertEqual(len(loaded.datasets), 2)
            self.assertEqual(len(loaded.fits), 1)
            self.assertEqual(loaded.fits[0]["uid"], "fit-uid-1")
            self.assertEqual(len(loaded.links), 1)

    def test_v3_deterministic_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_ordering")

            p = Project(
                name="ordering_test",
                description="Test",
                project_format_version=3,
            )
            p.datasets["z_dataset"] = {"uid": "z"}
            p.datasets["a_dataset"] = {"uid": "a"}
            p.datasets["m_dataset"] = {"uid": "m"}

            save_project(p, project_dir)

            with open(os.path.join(project_dir, "project.json")) as f:
                raw = json.load(f)

            dataset_keys = list(raw["datasets"].keys())
            self.assertEqual(dataset_keys, sorted(dataset_keys))

    def test_v3_always_outputs_v3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_v1")

            p = Project(
                name="v1_project",
                description="Legacy format",
                project_format_version=3,
            )
            p.datasets["ds1"] = {"path": "data/file1.dat"}

            project_json_path = save_project(p, project_dir)

            with project_json_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            self.assertEqual(raw["project_format_version"], 3)
            self.assertIn("meta", raw)

            loaded = load_project(project_dir)
            self.assertEqual(loaded.project_format_version, 3)
            self.assertEqual(loaded.name, "v1_project")

    def test_get_dataset_by_uid(self):
        p = Project(name="test", project_format_version=3)
        p.datasets["uid1"] = {"name": "dataset1"}
        p.datasets["uid2"] = {"name": "dataset2"}

        self.assertEqual(p.get_dataset("uid1")["name"], "dataset1")
        self.assertIsNone(p.get_dataset("nonexistent"))

    def test_get_fit_by_uid(self):
        p = Project(name="test", project_format_version=3)
        p.fits.append({"uid": "fit1", "name": "Fit 1"})
        p.fits.append({"uid": "fit2", "name": "Fit 2"})

        self.assertEqual(p.get_fit("fit1")["name"], "Fit 1")
        self.assertIsNone(p.get_fit("nonexistent"))

    def test_list_uids(self):
        p = Project(name="test", project_format_version=3)
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
