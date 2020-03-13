from __future__ import annotations

import weakref


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


