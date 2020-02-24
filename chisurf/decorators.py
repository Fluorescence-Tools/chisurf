from __future__ import annotations
import typing

import weakref
import os
import inspect
from qtpy import QtWidgets

import chisurf.widgets


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
            chisurf.widgets.load_ui(
                target=target,
                path=path,
                ui_filename=self.ui_filename
            )
            f(cls, *args, **kwargs)

        return wrapped


def register(cls):
    """Decorator to make a class a registered class.

    Example usage::

    @chisurf.decorators.register
    class A1():
        pass

    @chisurf.decorators.register
    class B():
        pass

    @chisurf.decorators.register
    class A2(A1):
        pass

    class A3(A1):
        pass

    a1_1 = A1()
    a1_2 = A1()
    a2_1 = A2()
    a3_1 = A3()
    b = B()

    assert a1_2 in a1_1.get_instances()
    assert a2_1 not in a1_1.get_instances()
    assert a3_1 in a1_1.get_instances()
    assert b not in a1_1.get_instances()

    """

    class RegisteredClass(cls):

        _instances = set()

        @classmethod
        def get_instances(
                cls
        ) -> weakref.ReferenceType:
            """Returns all instances of the class as an generator
            """
            dead = set()
            for ref in cls._instances:
                obj = ref()
                if obj is not None:
                    yield obj
                else:
                    dead.add(ref)
            cls._instances -= dead

        def __init__(
                self,
                *args,
                **kwargs
        ):
            self._instances.add(
                weakref.ref(self)
            )
            self.__class__.__name__ = cls.__name__
            # for name, member in self.__class__.__dict__.items():
            #     print(name)
            #     print(member)
            #     if not getattr(member, '__doc__'):
            #         self.__class__.__doc__ = getattr(cls, name).__doc__
            super().__init__(
                *args,
                **kwargs
            )

    return RegisteredClass


def set_module(module):
    """Decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
