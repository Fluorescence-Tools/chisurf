@property
def bytes(self):
    # type: () -> np.ndarray
    return self.get_bytes()


@bytes.setter
def bytes(self, v):
    # type: () -> (np.ndarray)
    self.set_bytes(v)


@property
def link(self):
    # type: () -> cn.Port
    return self.get_link()


@link.setter
def link(self, v):
    # type: (cn.Port) -> None
    if self.node is v.node:
        Warning("Linking to same Node.")
    self.set_link(v)


@property
def value(self):
    if self.get_value_type() == 0:
        v = self.get_value_i()
    elif self.get_value_type() == 1:
        v = self.get_value_d()
    elif self.get_value_type() == 2:
        v = self.get_value_i()[0]
    elif self.get_value_type() == 3:
        v = self.get_value_d()[0]
    else:
        v = None
    return v


@value.setter
def value(self, v):
    if not isinstance(v, np.ndarray):
        v = np.atleast_1d(v)
    if v.dtype.kind == 'i':
        self.set_value_i(v, True)
    else:
        self.set_value_d(v, True)


@property
def bounds(self):
    if self.bounded:
        return self.get_bounds()
    else:
        return None, None


@bounds.setter
def bounds(self, v):
    self.set_bounds(np.array(v, dtype=np.float64))


def __init__(
        self,
        value=[],
        fixed=False,
        *args, **kwargs
):
    this = _chinet.new_Port(*args, **kwargs)
    try:
        self.this.append(this)
    except:
        self.this = this
    if not isinstance(value, np.ndarray):
        value = np.atleast_1d(value)
    self.value = value
    self.fixed = fixed


def __repr__(self):
    return "Port(%s)" % self.oid


def __str__(self):
    return self.get_json(indent=4)

