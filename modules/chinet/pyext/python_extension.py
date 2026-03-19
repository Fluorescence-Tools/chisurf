import sys
if sys.version_info > (3, 0):
    from collections import UserDict
else:
    from UserDict import UserDict
import numpy as np
import chinet as cn
import inspect
import json
import types  # used by Node to identify code objects
import _chinet
__version__ = _chinet.CHINET_VERSION


def node(func):
    """A simple decorator to convert functions a chinet.Node object

    :param func:
    :return:
    """
    return cn.Node(
        callback_function=func,
        name=func.__name__,
    )


class NodeGroup(UserDict):

    name = None# type: str
    inputs = {} # type: dict
    outputs = {} # type: dict

    @property
    def is_valid(self):
        return not self.has_invalid_nodes()

    def __init__(
            self,
            name = '', # type: str
            nodes = None #type: dict
    ):
        self.name = name
        for nk in nodes:
            if not isinstance(nodes[nk], cn.Node):
                if isinstance(nodes[nk], dict):
                    nodes[nk] = cn.Node(**nodes[nk])
                elif isinstance(nodes[nk], list):
                    nodes[nk] = cn.Node(*nodes[nk])
                else:
                    raise ValueError("Node parameters need to be either list, dict, or cn.Node")
        super(NodeGroup, self).__init__(
            {
                'nodes': nodes
            }
        )

    def __len__(self):
        return len(self.data['nodes'])

    def __delitem__(self, key):
        del self.data['nodes'][key]

    def __iter__(self):
        return iter(self.data['nodes'])

    # Modify __contains__ to work correctly when __missing__ is present
    def __contains__(self, key):
        return key in self.data

    def __setitem__(self, key, value):
        if isinstance(value, cn.Node):
            self.add_ports_of_node(value)
            self.data['nodes'][key] = value
        else:
            super(NodeGroup, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key in self.data['nodes']:
            return self.data['nodes'][key]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __setattr__(self, name, value):
        try:
            self.inputs[name].value = value
        except KeyError:
            super(NodeGroup, self).__setattr__(name, value)

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

    def __call__(self, *args, **kwargs):
        self.evaluate()

    def add_ports_of_node(self, value):
        # type: (chinet.Node) -> None
        if isinstance(value, cn.Node):
            for k in value.inputs.keys():
                if k in self.inputs.keys():
                    Warning("Input key already exists. Linking to existing input ports")
                    if value.inputs[k] is not self.inputs[k]:
                        value.inputs[k].link = self.inputs[k]
                else:
                    self.inputs[k] = value.inputs[k]
            for k in value.outputs.keys():
                if k in self.outputs.keys():
                    Warning("Output port name %s already exists." % k)
                self.outputs[k] = value.outputs[k]

    def has_invalid_nodes(self):
        for nk in self:
            node = self[nk]
            if not node.is_valid:
                return True
        return False

    def evaluate(self):
        while self.has_invalid_nodes():
            for nk in self:
                n = self[nk]
                if not n.is_valid:
                    n.evaluate()


