from __future__ import annotations


def nusiance(
        f,
        *args,
        **kwargs
):
    """Scatter-corrected anisotropy

    :param Ss: perpenticular signal
    :param Sp: parallel signal
    :param Bs: perpenticular background
    :param Bp: parallel background
    :param Gfactor: g-factor
    :param l1: Objective mixing correction factor
    :param l2: Objective mixing correction factor
    :return: Anisotropy
    """

    def m(*args, **kwargs):

        # Anisotropy
        f.func_globals['Gfactor'] = kwargs.get('Gfactor', 1.0)
        f.func_globals['Bp'] = kwargs.get('Bp', 0.0)
        f.func_globals['Bs'] = kwargs.get('Bs', 0.0)
        f.func_globals['l1'] = kwargs.get('l1', 0.0)
        f.func_globals['l2'] = kwargs.get('l2', 0.0)

        # Intensities
        f.func_globals['Bg'] = kwargs.get('Bg', 0.0)
        f.func_globals['Br'] = kwargs.get('Br', 0.0)
        f.func_globals['crosstalk'] = kwargs.get('crosstalk', 0.0)

        f.func_globals['phiA'] = kwargs.get('phiA', 1.0)
        f.func_globals['phiD'] = kwargs.get('phiD', 1.0)

        f.func_globals['R0'] = kwargs.get('R0', 52.0)

        return f(*args, **kwargs)

    return m


