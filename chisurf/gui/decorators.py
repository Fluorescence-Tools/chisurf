import inspect
import os
from chisurf import typing

from qtpy import QtWidgets, uic


class init_with_ui(object):
    """
    This is a decorator for __init__ methods of QtWidget objects.
    The decorator accepts a ui_filename, calls the super class of
    the object, and initializes the ui file to a target specified
    by the target option. If no target is provided the ui file
    is initialized into the object of the __init__ function.

    """

    def __init__(
            self,
            ui_filename: str,
            path: str = None
    ):
        """

        :param ui_filename: The filename (without path) of the ui file.
        It is assumed that the ui file is in the same path as the file
        of the class that is being initialized.

        """
        self.ui_filename = ui_filename
        self.path = path

    def __call__(
            self,
            f: typing.Callable
    ):

        def load_ui(
                target: QtWidgets.QWidget,
                ui_filename: str,
                path: str
        ):
            uic.loadUi(
                os.path.join(
                    path,
                    ui_filename
                ),
                target
            )

        def wrapped(
                cls: QtWidgets.QWidget,
                *args,
                **kwargs
        ):
            if self.path is None:
                path = os.path.dirname(
                    inspect.getfile(
                        cls.__class__
                    )
                )
            else:
                path = self.path
            try:
                super(cls.__class__, cls).__init__(
                    *args,
                    **kwargs
                )
            except TypeError:
                super(cls.__class__, cls).__init__()

            target = cls
            load_ui(
                target=target,
                path=path,
                ui_filename=self.ui_filename
            )
            f(cls, *args, **kwargs)

        return wrapped