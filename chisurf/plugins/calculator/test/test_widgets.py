class TestFRETCalculator:
    def test_creation(self, qapp):
        from chisurf.plugins.calculator.fret_calculator import FRETCalculator
        widget = FRETCalculator()
        assert widget is not None


class TestHomoFRETCalculator:
    def test_creation(self, qapp):
        from chisurf.plugins.calculator.homofret_calculator import HomoFRETCalculator
        widget = HomoFRETCalculator()
        assert widget is not None
