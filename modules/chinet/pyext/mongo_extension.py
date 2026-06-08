
def __str__(self):
    return self.get_json(indent=4)


def __repr__(self):
    s = "MongoObject(%s)" % self.oid
    return s


def __init__(self, *args, **kwargs):
    this = _chinet.new_MongoObject(*args, **kwargs)
    try:
        self.this.append(this)
    except:
        self.this = this
    self.register_instance(None)


def __del__(self):
    self.unregister_instance(None)

