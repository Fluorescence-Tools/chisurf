"""JSON-RPC-compatible database connector services."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from chisurf.core.mfdb.database_resolver import (
    backup_database,
    resolve_database_path,
    source_database_path,
    user_database_path,
)
from chisurf.core.mfdb.importer import import_structure_file
from chisurf.core.mfdb.repository import MFDatabase


class DatabaseConnector:
    """Own the active sample database connection for core plugins."""

    def __init__(self) -> None:
        """Create an empty connector."""
        self._db: MFDatabase | None = None

    def open(self, database_path: str | None = None) -> dict[str, Any]:
        """Open the user database and return connector status.

        Parameters
        ----------
        database_path : str, optional
            Optional database path used for tests or advanced callers. If
            omitted, the standard source/user database resolver is used.

        Returns
        -------
        dict
            Connector status including the active database path.
        """
        path = Path(database_path) if database_path else resolve_database_path()
        if self._db is not None:
            self._db.close()
        self._db = MFDatabase(path)
        return self.status()

    def close(self) -> dict[str, Any]:
        """Close the active database connection.

        Returns
        -------
        dict
            Status after closing.
        """
        was_open = self._db is not None
        if self._db is not None:
            self._db.close()
            self._db = None
        return {"connected": False, "was_connected": was_open}

    def status(self) -> dict[str, Any]:
        """Return database connector status.

        Returns
        -------
        dict
            Source/user paths, schema version, and table counts.
        """
        path = Path(self._db.db_path) if self._db is not None else resolve_database_path()
        with MFDatabase(path) as db:
            experiment_rows = db.get_experiments()
            return {
                "source_database": str(source_database_path()),
                "user_database": str(path),
                "active_database": str(path),
                "schema_version": db._get_schema_version(),
                "sample_count": len(db.list_samples()),
                "user_count": len(db.get_users()),
                "device_count": len(db.get_devices()),
                "experiment_type_count": len(db.get_experiment_types()),
                "experiment_count": len(experiment_rows),
                "experiment_data_count": sum(len(db.get_experiment_data(row["experiment_id"])) for row in experiment_rows),
            }

    def backup(self) -> dict[str, Any]:
        """Back up the active user database.

        Returns
        -------
        dict
            Backup file path.
        """
        path = backup_database(resolve_database_path())
        return {"backup_path": str(path)}

    def reset_from_source(self) -> dict[str, Any]:
        """Replace the user database with the curated source database.

        Returns
        -------
        dict
            Reset status and optional backup path.
        """
        user_path = user_database_path()
        source_path = source_database_path()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        backup_path = backup_database(user_path) if user_path.exists() else None
        tmp_path = user_path.with_suffix(user_path.suffix + ".tmp")
        try:
            shutil.copy2(source_path, tmp_path)
            tmp_path.replace(user_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        if self._db is not None:
            self._db.close()
            self._db = None
        return {"ok": True, "backup_path": str(backup_path) if backup_path else None}

    def repository(self, include_counts: bool = True) -> dict[str, Any]:
        """Return repository metadata for the active database.

        Parameters
        ----------
        include_counts : bool, optional
            Include sample/user/device counts.

        Returns
        -------
        dict
            Repository metadata.
        """
        path = Path(self._db.db_path) if self._db is not None else resolve_database_path()
        with MFDatabase(path) as db:
            data: dict[str, Any] = {
                "database_path": str(path),
                "schema_version": db._get_schema_version(),
            }
            if include_counts:
                experiment_rows = db.get_experiments()
                data.update(
                    {
                        "sample_count": len(db.list_samples()),
                        "user_count": len(db.get_users()),
                        "device_count": len(db.get_devices()),
                        "experiment_type_count": len(db.get_experiment_types()),
                        "experiment_count": len(experiment_rows),
                        "experiment_data_count": sum(
                            len(db.get_experiment_data(row["experiment_id"]))
                            for row in experiment_rows
                        ),
                    }
                )
            return data

    def import_file(self, path: str) -> dict[str, Any]:
        """Import a structure file into the active database.

        Parameters
        ----------
        path : str
            PDBx/mmCIF, PDB-IHM, or FLR CIF file path.

        Returns
        -------
        dict
            Import summary.
        """
        with MFDatabase(resolve_database_path()) as db:
            summary = import_structure_file(db, path)
        return {"summary": summary}

    def export_sample(
        self,
        sample_id: str,
        output_path: str | None = None,
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        """Export a sample as FLR CIF.

        Parameters
        ----------
        sample_id : str
            Sample identifier.
        output_path : str, optional
            Optional output file path.
        analysis_id : str, optional
            FLR analysis identifier to export.

        Returns
        -------
        dict
            Output path or CIF text.
        """
        with MFDatabase(resolve_database_path()) as db:
            if analysis_id is None:
                row = db.conn.execute(
                    "SELECT analysis_id FROM flr_fret_analysis "
                    "WHERE sample_id = ? ORDER BY analysis_id LIMIT 1",
                    (sample_id,),
                ).fetchone()
                analysis_id = row["analysis_id"] if row else None
            if output_path:
                path = db.export_flr_cif(Path(output_path), analysis_id=analysis_id)
                return {"output_path": str(path)}
            return {"text": db.export_flr_cif_to_text(analysis_id=analysis_id)}


_connector = DatabaseConnector()


def register_services(dispatcher: Any) -> None:
    """Register database connector RPC handlers."""
    dispatcher.register("database_connector.status", lambda params: status_handler())
    dispatcher.register("database_connector.open", lambda params: open_handler(**params))
    dispatcher.register("database_connector.close", lambda params: close_handler())
    dispatcher.register("database_connector.backup", lambda params: backup_handler())
    dispatcher.register(
        "database_connector.reset_from_source", lambda params: reset_from_source_handler()
    )
    dispatcher.register(
        "database_connector.repository", lambda params: repository_handler(**params)
    )
    dispatcher.register("database_connector.import_file", lambda params: import_file_handler(**params))
    dispatcher.register("database_connector.export_sample", lambda params: export_sample_handler(**params))


def status_handler() -> dict[str, Any]:
    """Return connector status."""
    return _connector.status()


def open_handler(database_path: str | None = None) -> dict[str, Any]:
    """Open the database connector."""
    return _connector.open(database_path=database_path)


def close_handler() -> dict[str, Any]:
    """Close the database connector."""
    return _connector.close()


def backup_handler() -> dict[str, Any]:
    """Back up the active user database."""
    return _connector.backup()


def reset_from_source_handler() -> dict[str, Any]:
    """Reset the user database from the curated source database."""
    return _connector.reset_from_source()


def repository_handler(include_counts: bool = True) -> dict[str, Any]:
    """Return repository metadata."""
    return _connector.repository(include_counts=include_counts)


def import_file_handler(path: str) -> dict[str, Any]:
    """Import a structure file."""
    return _connector.import_file(path=path)


def export_sample_handler(
    sample_id: str,
    output_path: str | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    """Export a sample as FLR CIF."""
    return _connector.export_sample(
        sample_id=sample_id,
        output_path=output_path,
        analysis_id=analysis_id,
    )
