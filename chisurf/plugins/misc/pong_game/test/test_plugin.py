import importlib


def test_pong_plugin_import():
    module = importlib.import_module("chisurf.plugins.misc.pong_game")
    assert module is not None


def test_pong_plugin_init():
    module = importlib.import_module("chisurf.plugins.misc.pong_game")
    module.__name__ = "plugin"
