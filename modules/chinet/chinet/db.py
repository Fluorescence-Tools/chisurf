import json
import os

class DB:
    """
    In-memory singleton registry for chinet objects.
    Persistence is lazy and only occurs on explicit save/load.
    """
    _objects = {}

    @classmethod
    def register(cls, obj):
        if hasattr(obj, 'oid'):
            cls._objects[obj.oid] = obj
        elif isinstance(obj, dict) and "_id" in obj:
            cls._objects[obj["_id"]] = obj

    @classmethod
    def get(cls, oid):
        return cls._objects.get(oid)

    @classmethod
    def remove(cls, oid):
        if oid in cls._objects:
            del cls._objects[oid]

    @classmethod
    def clear(cls):
        cls._objects.clear()

    @classmethod
    def dump_all(cls):
        """Returns a list of all object documents."""
        data = []
        for obj in cls._objects.values():
            if hasattr(obj, '_document'):
                doc = obj._document.copy()
                if hasattr(obj, '_update_doc_from_data'):
                    obj._update_doc_from_data(doc)
                data.append(doc)
            elif isinstance(obj, dict):
                data.append(obj.copy())
        return data
