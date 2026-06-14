import inspect
import numpy as np
from .base import BaseObject
from .port import Port
from .db import DB

class Node(BaseObject):
    """
    Enhanced Node class with port management and lazy evaluation.
    """
    def __init__(self, name="", ports=None, callback_class=None, **kwargs):
        super().__init__(name=name)
        
        self.ports = {} # Map string to shared_ptr<Port>
        self.in_ = {}
        self.out_ = {}
        self.callback = ""
        self.callback_type_string = ""
        self.callback_type = -1
        self.callback_class = callback_class
        self.node_valid_ = False
        
        if callback_class is not None:
            self.callback_type = 1
            
        if ports:
            self.set_ports(ports)

        # Handle legacy callback_function kwarg
        if "callback_function" in kwargs:
            self.set_python_callback_function(kwargs["callback_function"])

        self._document["type"] = "node"

    def set_ports(self, ports):
        for name, port in ports.items():
            port.set_name(name)
            self.add_port(name, port, port.is_output, False)
        self.fill_input_output_port_lookups()

    def add_port(self, key, port, is_output, fill_in_out=True):
        port.set_port_type(is_output)
        port.set_node(self)
        self.ports[key] = port
        if fill_in_out:
            self.fill_input_output_port_lookups()

    def add_input_port(self, key, port):
        self.add_port(key, port, False)

    def add_output_port(self, key, port):
        self.add_port(key, port, True)

    def fill_input_output_port_lookups(self):
        self.in_.clear()
        self.out_.clear()
        for k, p in self.ports.items():
            if p.is_output:
                self.out_[k] = p
            else:
                self.in_[k] = p

    def set_callback(self, callback, callback_type):
        if isinstance(callback, str):
            self.callback = callback
            self.callback_type_string = callback_type
            t = callback_type.upper()
            if t == "C":
                self.callback_type = 0
            elif t in ("CLASS", "CPP", "C++"):
                self.callback_type = 1
            else:
                self.callback_type = -1
        else:
            self.callback_class = callback
            self.callback_type = 1

    def set_python_callback_function(self, func):
        self.callback_class = func
        self.callback_type = 1
        # Auto-infer ports
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            default = 0.0
            if param.default is not inspect.Parameter.empty:
                default = param.default
            p = Port(name=name); p.value = default
            self.add_input_port(name, p)

        # Evaluate with default arguments to see if it returns a dictionary of named output ports
        default_args = {}
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                default_args[name] = param.default
            else:
                default_args[name] = 0.0

        try:
            res = func(**default_args)
        except Exception:
            res = None

        if isinstance(res, dict):
            for k in res.keys():
                self.add_output_port(k, Port(name=k, is_output=True))
        else:
            self.add_output_port("out_00", Port(name="out_00", is_output=True))

    def inputs_valid(self):
        for input_port in self.in_.values():
            if input_port.is_linked():
                output_port = input_port.get_link()
                output_node = output_port.get_node()
                if output_node is self: continue
                if output_node is not None and not output_node.is_valid():
                    return False
        return True

    def is_valid(self):
        if not self.in_: return True
        if not self.inputs_valid(): return False
        return self.node_valid_

    def set_valid(self, v):
        self.node_valid_ = v

    def evaluate(self):
        if self.callback_class is None and self.callback_type < 0:
            return

        # Port operators
        if self.callback_type == 0:
            keys = list(self.in_.keys())
            if len(keys) >= 2:
                v1 = self.in_[keys[0]].value
                v2 = self.in_[keys[1]].value
                if self.callback == "addition_double":
                    self.out_[self.name].value = v1 + v2
                elif self.callback == "addition_int":
                    self.out_[self.name].value = int(v1 + v2)
                elif self.callback == "multiply_double":
                    self.out_[self.name].value = v1 * v2
                elif self.callback == "multiply_int":
                    self.out_[self.name].value = int(v1 * v2)

        if self.callback_class is not None:
            if callable(self.callback_class) and not hasattr(self.callback_class, 'run'):
                args = {k: v.value for k, v in self.in_.items()}
                try:
                    res = self.callback_class(**args)
                    if isinstance(res, dict):
                        for k, val in res.items():
                            if k in self.out_: self.out_[k].value = val
                    elif isinstance(res, (tuple, list)) and len(res) > 1:
                        for i, val in enumerate(res):
                            oname = f"out_{i:02d}"
                            if oname in self.out_: self.out_[oname].value = val
                    else:
                        if "out_00" in self.out_: self.out_["out_00"].value = res
                except TypeError:
                    self.callback_class(self.in_, self.out_)
            elif hasattr(self.callback_class, 'run'):
                self.callback_class.run(self.in_, self.out_)

        for p in self.out_.values():
            n = p.get_node()
            if n and n is not self: n.set_valid(False)
        
        self.node_valid_ = True

    def update(self):
        for p in self.in_.values():
            if p.is_linked():
                source_port = p.get_link()
                source_node = source_port.get_node()
                if source_node and not source_node.is_valid():
                    source_node.update()
                p.value = source_port.value
        if not self.is_valid():
            self.evaluate()

    def get_input_ports(self): return self.in_
    def get_output_ports(self): return self.out_
    def get_ports(self): return self.ports
    def get_port(self, name): return self.ports.get(name)
    def get_input_port(self, name): return self.in_.get(name)
    def get_output_port(self, name): return self.out_.get(name)

    @property
    def inputs(self):
        return self.in_

    @property
    def outputs(self):
        return self.out_

    def _update_doc_from_data(self, doc):
        doc.update({
            "callback": self.callback,
            "callback_type": self.callback_type_string,
            "valid": self.node_valid_,
            "ports": {k: p.oid for k, p in self.ports.items()}
        })

    def set_document(self, doc):
        super().set_document(doc)
        self.callback = doc.get("callback", "")
        self.callback_type_string = doc.get("callback_type", "")
        self.set_callback(self.callback, self.callback_type_string)
        self.node_valid_ = doc.get("valid", False)
        # Ports restoration handled by Session.load
