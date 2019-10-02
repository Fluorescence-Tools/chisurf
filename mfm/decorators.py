# #
# # import warnings
import weakref

# import os
# from qtpy import uic
#
#
#
# def load_ui(ui_file):
#     """This is a decorator which can be used to mark functions
#     as deprecated. It will result in a warning being emitted
#     when the function is used."""
#
#     def new_func(func, *args, **kwargs):
#         super(func.__class__, ).__init__(*args, **kwargs)
#         uic.loadUi(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(func.__module__.__file__)),
#                 ui_file
#             ),
#             args[0]
#         )
#         return func(*args, **kwargs)
#
#     return new_func
#

#
# def deprecated(func):
#     """This is a decorator which can be used to mark functions
#     as deprecated. It will result in a warning being emitted
#     when the function is used."""
#     @functools.wraps(func)
#     def new_func(*args, **kwargs):
#         warnings.simplefilter('always', DeprecationWarning)  # turn off filter
#         warnings.warn("Call to deprecated function {}.".format(func.__name__),
#                       category=DeprecationWarning,
#                       stacklevel=2)
#         warnings.simplefilter('default', DeprecationWarning)  # reset filter
#         return func(*args, **kwargs)
#     return new_func
#
#


def register(cls):
    """Decorator to make a class a registered class.

    Example usage::

    @mfm.decorators.register
    class A1():
        pass

    @mfm.decorators.register
    class B():
        pass

    @mfm.decorators.register
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
        def get_instances(cls):
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
            super().__init__(*args, **kwargs)

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
