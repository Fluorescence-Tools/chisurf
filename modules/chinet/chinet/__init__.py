from .port import Port
from .node import Node
from .session import Session, DB
from .base import BaseObject
from ._version import __version__

DatabaseObject = BaseObject
MemoryObject = BaseObject

class MongoObject(BaseObject):
    pass

Data = BaseObject

def save_session(filename):
    s = Session()
    s.save(filename)

def load_session(filename):
    return Session.load(filename)

def node(func):
    return Node(callback_function=func, name=func.__name__)

session = Session()
