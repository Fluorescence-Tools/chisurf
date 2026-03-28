import uuid
import json
import numpy as np
from .db import DB

class BaseObject:
    """
    Base class for all chinet objects.
    Mirrors MemoryObject 1:1.
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
        self._connected = True
        DB.register(self)
        return True

    def disconnect_from_db(self):
        self._connected = False
        return True

    def connect_object_to_db(self, obj):
        if obj: return obj.connect_to_db()
        return False

    def read_from_db(self, oid):
        self.oid = oid
        stored = DB.get(oid)
        if stored:
            self.set_document(stored._document if hasattr(stored, '_document') else stored)
            return True
        return False

    def write_to_db(self):
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

    # Legacy attribute helpers
    def set_singleton_double(self, k, v): self._document[k] = float(v)
    def set_singleton_int(self, k, v): self._document[k] = int(v)
    def set_singleton_bool(self, k, v): self._document[k] = bool(v)
    def get_singleton_double(self, k): return float(self._document.get(k, 0.0))
    def get_singleton_int(self, k): return int(self._document.get(k, 0))
    def get_singleton_bool(self, k): return bool(self._document.get(k, False))
    def set_array_double(self, k, v): self._document[k] = [float(x) for x in v]
    def set_array_int(self, k, v): self._document[k] = [int(x) for x in v]
    def get_array_double(self, k): return tuple(self._document.get(k, []))
    def get_array_int(self, k): return tuple(self._document.get(k, []))

    # SWIG names
    def get_own_oid(self): return self.oid
    def set_own_oid(self, v): self.oid = v
    def get_name(self): return self.name
    def set_name(self, v): self.name = v

    def connect_to_db_mongo(self, *args, **kwargs):
        return self.connect_to_db(*args, **kwargs)

    def connect_object_to_db_mongo(self, obj):
        return self.connect_object_to_db(obj)

    def read_from_db_mongo(self, oid):
        return self.read_from_db(oid)

    def set_value_vector(self, v): 
        # For Port subclasses this might be overridden, 
        # but BaseObject needs it for some TestMemoryObject tests
        self._document["value"] = [x for x in v]
        
    def get_value_vector(self): 
        return self._document.get("value", [])

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
