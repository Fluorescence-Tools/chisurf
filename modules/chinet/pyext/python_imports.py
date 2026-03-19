import numpy as np
import chinet as cn
import json
import types  # used by Node to identify code objects
import _chinet
__version__ = _chinet.CHINET_VERSION


def node(func):
    """A simple decorator to convert functions a chinet.Node object

    :param func:
    :return:
    """
    return cn.Node(
        callback_function=func,
        name=func.__name__,
    )

