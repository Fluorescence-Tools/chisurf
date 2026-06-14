import json
import os

class DB:
    """
    In-memory singleton registry for chinet objects.

    Optional backend adapters may be configured for transparent durable storage,
    but the registry itself remains process-local.
    """
    _objects = {}
    _backend = None

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
    def set_backend(cls, backend):
        """Configure an optional transparent persistence backend."""
        cls._backend = backend

    @classmethod
    def clear_backend(cls):
        """Remove the optional transparent persistence backend."""
        cls._backend = None

    @classmethod
    def get_backend(cls):
        """Return the configured optional backend, if any."""
        return cls._backend

    @classmethod
    def has_backend(cls):
        """Return whether a transparent persistence backend is configured."""
        return cls._backend is not None

    @classmethod
    def iter_objects(cls):
        """Iterate over currently registered chinet objects."""
        return iter(cls._objects.values())

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
