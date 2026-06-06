class TestPCHApp:
    def test_creation(self, qapp):
        from chisurf.plugins.pch import PCHApp
        widget = PCHApp()
        assert widget is not None

    def test_window_title(self, qapp):
        from chisurf.plugins.pch import PCHApp
        widget = PCHApp()
        assert "PCH" in widget.windowTitle()
