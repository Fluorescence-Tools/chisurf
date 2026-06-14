import uuid
import json
import numpy as np
from .db import DB


def _is_mfdb_request(args, kwargs):
    """Return whether connect arguments request the optional MFDB backend."""
    if args and args[0] == "mfdb":
        return True
    return (
        kwargs.get("backend") == "mfdb"
        or str(kwargs.get("uri_string", "")).lower().startswith(("mfdb:", "mfdb://"))
    )


def _configure_backend_from_request(args, kwargs):
    """Configure and return a backend requested through connect_to_db."""
    backend = kwargs.get("backend")
    if backend == "mfdb":
        kwargs.pop("backend", None)
    if backend is not None and backend != "mfdb":
        DB.set_backend(backend)
        return backend
    if _is_mfdb_request(args, kwargs):
        from chisurf.core.mfdb.chinet_adapter import configure_mfdb_backend

        if args and args[0] == "mfdb":
            db_path = kwargs.pop("db_path", None)
            operation_id = kwargs.pop("operation_id", None)
            experiment_id = kwargs.pop("experiment_id", None)
            store_node_artifacts = kwargs.pop("store_node_artifacts", True)
            parameters = kwargs.pop("parameters", None)
            operation_type = kwargs.pop("operation_type", "model_fitting")
            metadata = kwargs.pop("metadata", None)
            backend = configure_mfdb_backend(
                db_path=db_path,
                operation_id=operation_id,
                experiment_id=experiment_id,
                store_node_artifacts=store_node_artifacts,
                parameters=parameters,
                operation_type=operation_type,
                metadata=metadata,
            )
        else:
            backend = configure_mfdb_backend(**kwargs)
        return backend
    return None


class BaseObject:
    """
    Base class for all chinet objects.
    """
    def __init__(self, name=""):
        self.object_name = name
        self.oid_document = str(uuid.uuid4())
        self.oid_precursor = self.oid_document
        self.time_of_death = 0
        self._connected = False
        self._document = {
            "_id": self.oid_document,
            "name": self.object_name,
            "type": self.__class__.__name__.lower(),
            "precursor": self.oid_precursor,
            "death": self.time_of_death
        }
        DB.register(self)

    @property
    def oid(self): return self.oid_document
    @oid.setter
    def oid(self, v): 
        self.oid_document = v
        self._document["_id"] = v

    @property
    def name(self): return self.object_name
    @name.setter
    def name(self, v): 
        self.object_name = v
        self._document["name"] = v

    @property
    def is_connected_to_db(self): return self._connected

    def connect_to_db(self, *args, **kwargs):
        backend = _configure_backend_from_request(args, kwargs)
        if backend is not None or DB.has_backend():
            DB.register(self)
            active_backend = backend or DB.get_backend()
            try:
                result = active_backend.connect_object(self, *args, **kwargs)
            except TypeError:
                if args or kwargs:
                    raise
                result = active_backend.connect_object(self)
                if result is None:
                    result = True
            self._connected = bool(result)
            return self._connected
        self._connected = True
        DB.register(self)
        return True

    def disconnect_from_db(self):
        if DB.has_backend():
            try:
                DB.get_backend().disconnect_object(self)
            except AttributeError:
                pass
        self._connected = False
        return True

    def connect_object_to_db(self, obj):
        if obj: return obj.connect_to_db()
        return False

    def read_from_db(self, oid):
        if DB.has_backend():
            try:
                return bool(DB.get_backend().read_object(self, oid))
            except AttributeError:
                return False
        self.oid = oid
        stored = DB.get(oid)
        if stored:
            self.set_document(stored._document if hasattr(stored, '_document') else stored)
            return True
        return False

    def write_to_db(self):
        if DB.has_backend():
            try:
                result = DB.get_backend().write_object(self)
            except AttributeError:
                result = False
            self._connected = bool(result)
            return result
        if not self._connected: return False
        doc = self._document
        if hasattr(self, '_update_doc_from_data'):
            self._update_doc_from_data(doc)
        DB.register(self)
        return True

    def set_document(self, doc):
        self._document = doc.copy()
        self.oid_document = doc.get("_id", self.oid_document)
        self.object_name = doc.get("name", self.object_name)
        self.time_of_death = doc.get("death", 0)

    def get_json(self, indent=0):
        doc = self._document.copy()
        if hasattr(self, '_update_doc_from_data'):
            self._update_doc_from_data(doc)
        return json.dumps(doc, indent=indent if indent > 0 else None, 
                          default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    def create_copy_in_db(self):
        new_obj = self.__class__(name=self.object_name + "_copy")
        new_obj.set_document(self._document)
        new_obj.oid = str(uuid.uuid4())
        new_obj.oid_precursor = self.oid_document  # Set precursor to parent
        new_obj._document["precursor"] = self.oid_document
        new_obj.connect_to_db()
        new_obj.write_to_db()
        return new_obj.oid

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', oid='{self.oid}')>"
