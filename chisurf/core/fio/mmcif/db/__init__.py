try:
    from .seed_data import seed_curated_database
except ModuleNotFoundError:
    def seed_curated_database(*args, **kwargs):
        """No-op fallback when curated seed data is not installed.

        Parameters
        ----------
        *args
            Positional arguments ignored by the fallback.
        **kwargs
            Keyword arguments ignored by the fallback.
        """
        return None
from .repository import FluorescenceDatabase, FluorophoreDatabase
from chisurf.core.mfdb import schema
try:
    from .importer import import_structure_file
except ModuleNotFoundError:
    def import_structure_file(*args, **kwargs):
        """Raise a clear error when the optional structure importer is absent.

        Parameters
        ----------
        *args
            Positional arguments ignored by the fallback.
        **kwargs
            Keyword arguments ignored by the fallback.
        """
        raise ModuleNotFoundError("chisurf.core.fio.mmcif.db.importer")
try:
    from .database_resolver import resolve_database_path, source_database_path, user_database_path, backup_database, backup_database_before_migration
except ModuleNotFoundError:
    resolve_database_path = source_database_path = user_database_path = None
    backup_database = backup_database_before_migration = None
try:
    from .zmq_server import FlrDatabaseServer
except ModuleNotFoundError:
    FlrDatabaseServer = None
try:
    from .zmq_client import FlrDatabaseClient
except ModuleNotFoundError:
    FlrDatabaseClient = None
from chisurf.core.mfdb.models import *
