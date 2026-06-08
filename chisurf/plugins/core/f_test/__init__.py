from chisurf.plugins.core.f_test.f_calculator import FTestWidget

name = "Main:Tools:F-Test"

if __name__ == "plugin":
    window = FTestWidget()
    window.show()
