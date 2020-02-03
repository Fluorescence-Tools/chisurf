import chisurf
import chisurf.experiments


def change_model(
        function_str: str
) -> None:
    cs = chisurf.cs
    for f in cs.current_fit:
        f.model.parse.func = "%s" % function_str
        f.model.parse.onEquationChanged()
