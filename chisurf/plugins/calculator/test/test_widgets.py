class TestFRETCalculator:
    def test_creation(self, qapp):
        from chisurf.plugins.calculator.fret_calculator import FretCalculatorTool
        from chisurf.plugins.calculator.fret_calculator.gui.tool import _FretTab

        widget = FretCalculatorTool()
        assert widget is not None
        # the hetero-FRET (DA) tab is rendered declaratively via AutoForm
        da_tab = widget.findChild(_FretTab)
        assert da_tab is not None
        assert da_tab.spin_tau0.suffix() == " ns"


class TestHomoFRETCalculator:
    def test_creation(self, qapp):
        from chisurf.plugins.calculator.fret_calculator import FretCalculatorTool
        from chisurf.plugins.calculator.fret_calculator.gui.tool import _HomoFretTab

        widget = FretCalculatorTool()
        assert widget is not None
        # the homo-FRET tab lives inside the same tabbed tool
        homo_tab = widget.findChild(_HomoFretTab)
        assert homo_tab is not None
        # k_homo is a read-only output field
        assert homo_tab.spin_kHomo.isReadOnly()
