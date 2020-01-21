import chisurf
import chisurf.experiments


def change_model(
        function_str: str
) -> None:
    cs = chisurf.cs
    for f in cs.current_fit:
        eval("cs.current_fit.model.parse.func = '%s'" % function_str)
        f.model.update()
