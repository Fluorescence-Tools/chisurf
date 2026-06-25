from ._version import __version__ as __version__
from .base import BaseObject as BaseObject
from .node import Node as Node
from .port import Port as Port
from .port import LinkCycleError as LinkCycleError
from .schema import (
    SCHEMA_NAME as SCHEMA_NAME,
)
from .schema import (
    SCHEMA_VERSION as SCHEMA_VERSION,
)
from .schema import (
    session_from_schema as session_from_schema,
)
from .schema import (
    session_to_schema as session_to_schema,
)
from .session import DB as DB
from .session import Session as Session

Data = BaseObject

__all__ = [
    "BaseObject",
    "DB",
    "Data",
    "Node",
    "Port",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "Session",
    "__version__",
    "load_session",
    "node",
    "save_session",
    "session",
    "session_from_schema",
    "session_to_schema",
]


def save_session(filename):
    s = Session()
    s.save(filename)


def load_session(filename):
    return Session.load(filename)


def node(func):
    return Node(callback_function=func, name=func.__name__)


session = Session()
