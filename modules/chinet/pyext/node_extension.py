@property
def inputs(self):
    return dict(self.get_input_ports())


@property
def outputs(self):
    return dict(self.get_output_ports())


@property
def ports(self):
    return self.get_ports()


@ports.setter
def ports(self, v):
    # type: (Dict[str, object]) -> None
    p = dict()
    for key in v:
        if isinstance(v[key], cn.Port):
            p[key] = v[key]
        elif isinstance(v[key], dict):
            p[key] = cn.Port(**v[key])
        else:
            p[key] = cn.Port(value=v[key])
    self.set_ports(p)


def set_python_callback_function(
        self,
        cb,
        reactive_inputs = True,
        reactive_outputs = True
):
    # type: (Callable, bool, bool) -> None
    class CallbackNodePython(cn.NodeCallback):

        def __init__(
                self,
                cb_function, # type: Callable
                *args, **kwargs
        ):
            super(CallbackNodePython, self).__init__(*args, **kwargs)
            self._cb = cb_function

        def run(
                self,
                inputs, # type: List[cn.Port]
                outputs # type: List[cn.Port]
        ):
            input_dict = dict(
                [(inputs[i].name, inputs[i].value) for i in inputs]
            )
            output = self._cb(**input_dict)
            if isinstance(output, dict):
                for key, ov in zip(outputs, output):
                    outputs[key].value = output[key]
            elif isinstance(output, tuple):
                for key, ov in zip(outputs, output):
                    outputs[key].value = ov
            else:
                next(iter(outputs.values())).value = output

    if isinstance(cb, str):
        code_obj = compile(cb, '<string>', 'exec')
        self.set_string('source', cb)
        for o in code_obj.co_consts:
            if isinstance(o, types.CodeType):
                cb = types.FunctionType(o, globals())
                break
    else:
        self.set_string('source', inspect.getsource(cb))
    # When the cb-function of a node is evaluated, the input Ports are
    # processed by the callback and written to the output Port. Thus,
    # the inputs and outputs need to be defined as Ports. Here, the inputs
    # and the output of the function are inspected to create the Ports of the
    # None.

    # get the signature of the function to create input Ports
    import sys    
    if sys.version_info[0] > 2:
        from inspect import signature as sig
        empty_name = inspect.Signature.empty
        empty_type = inspect._empty
        _use_py2 = True
    else:
        import funcsigs
        from funcsigs import signature as sig
        empty_name = funcsigs._empty
        empty_type = funcsigs._empty
        _use_py2 = False

    _sig = sig(cb)
    input_ports = list()
    for parameter_name in _sig.parameters:
        o = _sig.parameters[parameter_name]
        # Check the function signature to use default values as cn.Port value
        if o.default is empty_name:
            if o.annotation is empty_type:
                print("WARNING no type specified for %s cn.Port." % parameter_name)
                value = np.array([1.0], dtype=np.float64)
            elif o.annotation == 'int':
                value = 1
            elif 'ndarray' == o.annotation:
                value = np.array([1.0], dtype=np.float64)
            elif o.annotation == 'float':
                value = 1.0
            else:
                print("WARNING no type specified for %s cn.Port." % parameter_name)
                value = np.array([1.0], dtype=np.float64)
        else:
            value = o.default
        if o.name not in self.ports.keys():
            p = cn.Port(
                name=o.name,
                value=value,
                is_output=False,
                is_reactive=reactive_inputs
            )
            self.add_input_port(
                key=o.name,
                port=p
            )
            input_ports.append(p)
        else:
            input_ports.append(self.get_input_ports()[o.name])
    # run the function once with the default values to get the output
    input_values = dict([(p.name, p.value) for p in input_ports])
    returned_value = cb(**input_values)
    # use the return value of the function to create output Ports
    output_values = list()
    output_names = list()
    if isinstance(returned_value, dict):
        for ok in returned_value:
            output_values.append(returned_value[ok])
            output_names.append(ok)
    elif isinstance(returned_value, tuple):
        output_values = returned_value
        for i in range(len(returned_value)):
            output_names.append("out_" + str(i).zfill(2))
    else:
        output_names = 'out_00',
        output_values = returned_value,
    for on, ov in zip(output_names, output_values):
        if on not in self.get_output_ports().keys():
            self.add_output_port(
                key=on,
                port=cn.Port(
                    name=on,
                    value=ov,
                    is_output=True,
                    is_reactive=reactive_outputs
                )
            )
    # create a new CallbackNodePython
    cb_instance = CallbackNodePython(cb_function=cb)
    cb_instance.__disown__()
    self.set_callback(cb_instance)


callback_function = property(None, set_python_callback_function)


def __getattr__(self, name):
    in_input = name in self.inputs.keys()
    in_output = name in self.outputs.keys()
    if in_input and in_output:
        raise KeyError("Ambiguous access. %s an input and output parameter.")
    elif not in_output and not in_input:
        raise AttributeError
    else:
        if in_input:
            return self.inputs[name].value
        else:
            return self.outputs[name].value


def __setattr__(self, name, value):
    try:
        in_input = name in self.inputs.keys()
        in_output = name in self.outputs.keys()
        if in_input and in_output:
            raise KeyError("Ambiguous access. %s an input and output parameter.")
        elif not in_output and not in_input:
            raise AttributeError
        else:
            if in_input:
                self.inputs[name].value = value
            else:
                self.outputs[name].value = value
    except:
        # super().__setattr__(name, value)
        MongoObject.__setattr__(self, name, value)


def __call__(self):
    return self.evaluate()


def __del__(self):
    # super(Node).__del__()
    MongoObject.__del__(self)


def __init__(self,
             obj=None,
             name = "", #type: str
             ports = None, #type: dict
             callback_function=None,
             reactive_inputs=True,
             reactive_outputs=True,
             *args, **kwargs
):
    """

    :param self:
    :param obj: Either the name (if string) or the callback function (if callable)
    :param name: The name of the Nose
    :param ports:
    :param callback_function: The callback function of the Node object
    :param reactive_inputs: if True sets all inputs that are created when the
    object is created to reactive
    :param args:
    :param kwargs:
    :return:
    """
    this = _chinet.new_Node(*args, **kwargs)
    try:
        self.this.append(this)
    except:
        self.this = this
    self.register_instance(None)
    if isinstance(ports, dict):
        self.ports = ports
    if isinstance(obj, str):
        name = obj
    elif callable(obj):
        callback_function = obj
    # set the python callback function
    if callable(callback_function) or isinstance(callback_function, str):
        self.set_python_callback_function(
            cb=callback_function,
            reactive_inputs=reactive_inputs,
            reactive_outputs=reactive_outputs
        )
    if len(name) > 0:
        self.name = name
    else:
        if callable(callback_function):
            self.name = callback_function.__name__
