
def __str__(self):
    return self.get_json(indent=4)


def __repr__(self):
    # Use the actual class name (MongoObject or MemoryObject) instead of DatabaseObject
    class_name = self.__class__.__name__
    s = "%s(%s)" % (class_name, self.oid)
    return s


@property
def is_connected_to_db(self):
    # Call the is_connected_to_db method directly through _chinet to avoid recursion
    class_name = self.__class__.__name__
    if class_name == "MongoObject":
        return _chinet.MongoObject_is_connected_to_db(self)
    else:  # MemoryObject
        return _chinet.MemoryObject_is_connected_to_db(self)


def __init__(self, *args, **kwargs):
    # The actual class will be either MongoObject or MemoryObject
    # depending on whether WITH_MONGODB is defined
    class_name = self.__class__.__name__
    if class_name == "MongoObject":
        this = _chinet.new_MongoObject(*args, **kwargs)
    else:  # MemoryObject
        this = _chinet.new_MemoryObject(*args, **kwargs)
    try:
        self.this.append(this)
    except:
        self.this = this
    self.register_instance(None)


def __del__(self):
    try:
        self.unregister_instance(None)
    except TypeError:
        # If the object is already being destroyed, the unregister_instance method might fail
        # with a TypeError because 'self' is no longer a valid MemoryObject/MongoObject pointer
        pass
