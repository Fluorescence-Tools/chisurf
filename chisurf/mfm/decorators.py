import warnings
import weakref
from functools import wraps


def deprecated(func):
    """ This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


'''
def register(*args, **kwargs):

    def class_rebuilder(cls):

        #@wraps(cls)
        class RegisteredClass(cls):

            _instances = kwargs.get('instances', set())

            @classmethod
            def getinstances(cls):
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

            def __init__(self, *args, **kwargs):
                self._instances.add(weakref.ref(self))
                #super(cls.mro()[1], self).__init__(*args, **kwargs)
                #RegisteredClass.mro()[1].__init__(self, *args, **kwargs)
                super(RegisteredClass, self).__init__(*args, **kwargs)
                #super(cls, self).__init__(*args, **kwargs)
                #cls.__init__(self, *args, **kwargs)

            def __getattribute__(self, attr_name):
                obj = super(RegisteredClass, self).__getattribute__(attr_name)
                if hasattr(obj, '__call__') and attr_name == '_instances':
                    return self._instances
                return obj

            #def __repr__(self):
            #    return super(RegisteredClass, self).__repr__()

        try:
            RegisteredClass.__doc__ = cls.__doc__
        except AttributeError:
            pass
        return RegisteredClass

    return class_rebuilder
'''
