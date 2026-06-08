class TestKappa2Dist:
    def test_creation(self, qapp, qtbot):
        from chisurf.plugins.kappa2_dist import Kappa2Dist
        widget = Kappa2Dist()
        qtbot.addWidget(widget)
        assert widget is not None

    def test_creation_with_kappa2(self, qapp, qtbot):
        from chisurf.plugins.kappa2_dist import Kappa2Dist
        widget = Kappa2Dist(kappa2=0.667)
        qtbot.addWidget(widget)
        assert widget is not None
        assert widget.kappa2 == 0.667
