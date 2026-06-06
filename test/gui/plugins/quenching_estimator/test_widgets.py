class TestQuEstWindow:
    def test_creation(self, qapp):
        from chisurf.plugins.quenching_estimator import QuEstWindow
        widget = QuEstWindow()
        assert widget is not None
