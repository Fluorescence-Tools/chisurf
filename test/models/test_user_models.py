import pathlib
import unittest

import utils

TOPDIR = pathlib.Path(__file__).parent.parent
utils.set_search_paths(TOPDIR)

import chisurf.core.models


class TestUserModels(unittest.TestCase):

    def setUp(self):
        chisurf.core.models._user_models_loaded = False
        chisurf.core.models._user_model_registry.clear()

    def test_register_user_model_decorator(self):
        @chisurf.core.models.register_user_model(["tcspc", "TCSPC"])
        class DummyTCSPCModel(chisurf.core.models.Model):
            name = "DummyTCSPCModel"

            def update_model(self, **kwargs):
                pass

            def update(self, **kwargs) -> None:
                pass

        models = list(
            chisurf.core.models.iter_user_models_for_experiment(
                "tcspc",
                "TCSPC",
            )
        )
        self.assertIn(DummyTCSPCModel, models)

    def test_load_user_models_from_settings_folder(self):
        from test import utils as test_utils

        with test_utils.temporary_directory() as tmpdir:
            settings_dir = pathlib.Path(tmpdir)
            models_dir = settings_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            module_path = models_dir / "user_tcspc_model.py"
            module_source = (
                "import chisurf\n"
                "import chisurf.core.models\n\n"
                "@chisurf.core.models.register_user_model(['tcspc'])\n"
                "class FileTCSPCModel(chisurf.core.models.Model):\n"
                "    name = 'FileTCSPCModel'\n\n"
                "    def update_model(self, **kwargs):\n"
                "        pass\n\n"
                "    def update(self, **kwargs) -> None:\n"
                "        pass\n"
            )
            module_path.write_text(module_source, encoding="utf-8")

            chisurf.core.models._user_models_loaded = False
            chisurf.core.models._user_model_registry.clear()

            original_get_path = chisurf.core.models.get_path
            try:
                chisurf.core.models.get_path = lambda kind="settings": settings_dir
                chisurf.core.models.load_user_models()
            finally:
                chisurf.core.models.get_path = original_get_path

            models = list(
                chisurf.core.models.iter_user_models_for_experiment(
                    "tcspc",
                    "TCSPC",
                )
            )
            names = {m.__name__ for m in models}
            self.assertIn("FileTCSPCModel", names)
