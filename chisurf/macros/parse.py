"""

"""
import chisurf
import chisurf.experiments


def change_model(
        function_str: str
) -> None:
    """

    :param function_str:
    :return:
    """
    cs = chisurf.cs
    fit = cs.current_fit
    for f in fit:
        f.model.func = "%s" % function_str
    # fit.model.parse.onEquationChanged()
