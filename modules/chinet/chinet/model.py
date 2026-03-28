import numpy as np
from .port import Port, HAS_IMP
from .utils import is_chinet_verbose

if HAS_IMP:
    import IMP
    import IMP.core

class Node:
    """
    chinet Node that manages Ports and maps to an IMP Hierarchy.
    """
    
    def __init__(self, name="node", model=None):
        self._name = name
        self._model = model
        self._inputs = {}
        self._outputs = {}
        
        self._particle = None
        if HAS_IMP and model is not None:
            self._particle = IMP.Particle(model, name)
            IMP.core.Hierarchy.setup_particle(self._particle)
            
    def add_input(self, name, value=0.0):
        p = Port(value=value, name=name, model=self._model)
        self._inputs[name] = p
        if self._particle:
            IMP.core.Hierarchy(self._particle).add_child(p.particle)
        return p

    def add_output(self, name, value=0.0):
        p = Port(value=value, name=name, model=self._model)
        self._outputs[name] = p
        if self._particle:
            IMP.core.Hierarchy(self._particle).add_child(p.particle)
        return p

    def evaluate(self):
        """User defined evaluation logic."""
        pass

    def update(self):
        """Syncs ports and evaluates."""
        if is_chinet_verbose():
            print(f"Updating node {self._name}")
        for p in self._inputs.values():
            p.sync_from_imp()
        
        self.evaluate()
        
        for p in self._outputs.values():
            p.sync_to_imp()

class Session:
    """
    chinet Session wrapping an IMP Model.
    """
    
    def __init__(self):
        if HAS_IMP:
            self._model = IMP.Model()
        else:
            self._model = None
        self._nodes = {}

    def create_node(self, name):
        node = Node(name=name, model=self._model)
        self._nodes[name] = node
        return node

    @property
    def model(self):
        return self._model
