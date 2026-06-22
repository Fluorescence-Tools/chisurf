import json as _json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

SCHEMA_VERSION = 39


@dataclass
class MigrationReport:
    """Structured report of a schema migration and backfill operation.

    Parameters
    ----------
    from_version : int
        Schema version before migration.
    to_version : int
        Schema version after migration.
    tables_added : list of str
        Table names added during migration.
    backfill : dict of str -> dict
        Per-table backfill counts with keys ``source``, ``inserted``,
        ``skipped``, and optional ``warnings``.
    """

    from_version: int
    to_version: int
    tables_added: list[str] = field(default_factory=list)
    backfill: dict[str, dict[str, int | list[str]]] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        lines = [
            f"Schema migrated from v{self.from_version} to v{self.to_version}",
        ]
        if self.tables_added:
            lines.append(f"  Tables added: {', '.join(self.tables_added)}")
        for table, counts in self.backfill.items():
            lines.append(f"  {table}: {counts.get('source', 0)} source, "
                         f"{counts.get('inserted', 0)} inserted, "
                         f"{counts.get('skipped', 0)} skipped")
            warnings = counts.get("warnings", [])
            if warnings:
                lines.append(f"    warnings ({len(warnings)}): {warnings[:5]}")
        return "\n".join(lines)


def _get_dict_ddl(category_name: str) -> str:
    """Lazy-cached DDL for a dictionary category.  Used by ``migrate_schema``."""
    cache = _get_dict_ddl.__dict__.setdefault("_cache", {})
    if category_name not in cache:
        from chisurf.core.mfdb.schema_from_dictionary import (
            generate_create_table_for_category,
        )
        from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
        cache[category_name] = generate_create_table_for_category(
            MmcifDictionary.load_bundled(), category_name
        )
    return cache[category_name]


CREATE_TABLES_SQL = [
    "CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER)",
    "CREATE TABLE IF NOT EXISTS probe_types (type_id INTEGER PRIMARY KEY, type_name TEXT UNIQUE, display_name TEXT)",
    """CREATE TABLE IF NOT EXISTS chem_descriptors (
        id INTEGER PRIMARY KEY, descriptor_type TEXT, descriptor TEXT,
        program TEXT, program_version TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS probes (
        probe_id INTEGER PRIMARY KEY,
        chromophore_name TEXT NOT NULL,
        reactive_probe_flag TEXT DEFAULT 'no',
        reactive_probe_name TEXT,
        probe_origin TEXT DEFAULT 'extrinsic',
        probe_link_type TEXT DEFAULT 'covalent',
        fluorophore_type TEXT DEFAULT 'unspecified',
        chromophore_chem_descriptor_id INTEGER REFERENCES chem_descriptors(id),
        reactive_probe_chem_descriptor_id INTEGER REFERENCES chem_descriptors(id),
        chromophore_center_atom TEXT,
        description TEXT DEFAULT '',
        category TEXT DEFAULT 'other',
        is_curated INTEGER DEFAULT 0,
        quality_flag INTEGER DEFAULT 1,
        type_id INTEGER REFERENCES probe_types(type_id),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS entities (
        entity_id TEXT PRIMARY KEY,
        type TEXT DEFAULT 'polymer',
        description TEXT,
        formula_weight REAL,
        src_method TEXT,
        number_of_molecules INTEGER DEFAULT 1,
        common_name TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS entity_poly_seq (
        id INTEGER PRIMARY KEY,
        entity_id TEXT NOT NULL REFERENCES entities(entity_id),
        num INTEGER NOT NULL,
        mon_id TEXT NOT NULL,
        hetero TEXT DEFAULT 'n',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (entity_id, num)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample (
        sample_id TEXT PRIMARY KEY,
        description TEXT,
        details TEXT,
        sample_type TEXT,
        num_of_probes INTEGER,
        solvent_phase TEXT,
        sample_condition_id TEXT,
        entity_assembly_id TEXT,
        project_id TEXT,
        measured_by_user_id TEXT,
        measured_by_device_id TEXT,
        measured_at TEXT,
        sample_uuid TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_condition (
        condition_id TEXT PRIMARY KEY,
        ph REAL,
        temperature REAL,
        ionic_strength REAL,
        buffer_composition TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_users (
        user_id TEXT PRIMARY KEY,
        user_uuid TEXT UNIQUE,
        display_name TEXT NOT NULL,
        email TEXT,
        affiliation TEXT,
        department TEXT,
        role TEXT,
        address TEXT,
        website TEXT,
        phone TEXT,
        is_admin INTEGER DEFAULT 0,
        allow_passwordless_login INTEGER DEFAULT 0,
        password_hash TEXT,
        details TEXT,
        active_branch_uuid TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_devices (
        device_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        device_type TEXT,
        model TEXT,
        serial_number TEXT,
        location TEXT,
        owner TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_experiment_type (
        type_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        category TEXT,
        description TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_experiment (
        experiment_id TEXT PRIMARY KEY,
        type_id INTEGER REFERENCES flr_experiment_type(type_id),
        sample_id TEXT REFERENCES flr_sample(sample_id),
        project_id TEXT,
        measured_by_user_id TEXT,
        measured_by_device_id TEXT,
        started_at TEXT,
        ended_at TEXT,
        status TEXT,
        details TEXT,
        setup_definition_id TEXT REFERENCES mfdb_setup(setup_id) ON DELETE SET NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_experiment_key_value (
        experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (experiment_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_experiment_data (
        data_id INTEGER PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        data_type TEXT NOT NULL,
        storage_mode TEXT NOT NULL,
        file_path TEXT,
        url TEXT,
        folder_path TEXT,
        mime_type TEXT,
        size_bytes INTEGER,
        checksum TEXT,
        data_json TEXT,
        data_blob BLOB,
        reading_options_json TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_key_value (
        sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (sample_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_probe (
        sample_probe_id INTEGER PRIMARY KEY,
        sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        poly_probe_position_id INTEGER REFERENCES flr_poly_probe_position(id),
        fluorophore_type TEXT DEFAULT 'unspecified',
        description TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (sample_id, probe_id, poly_probe_position_id)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_poly_probe_position (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        entity_id TEXT NOT NULL REFERENCES entities(entity_id),
        asym_id TEXT DEFAULT 'A',
        residue_number INTEGER NOT NULL,
        residue_name TEXT,
        atom_id TEXT,
        mutation_flag TEXT DEFAULT 'no',
        modification_flag TEXT DEFAULT 'no',
        auth_name TEXT,
        description TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (probe_id, entity_id, asym_id, residue_number)
    )""",
    """CREATE TABLE IF NOT EXISTS optical_properties (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        property_name TEXT NOT NULL,
        property_value TEXT,
        unit TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (probe_id, property_name)
    )""",
    """CREATE TABLE IF NOT EXISTS spectra (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        spectrum_type TEXT NOT NULL,
        wavelengths BLOB NOT NULL,
        intensity_values BLOB NOT NULL,
        wavelength_unit TEXT DEFAULT 'nm',
        intensity_unit TEXT DEFAULT 'normalized',
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (probe_id, spectrum_type)
    )""",
    """CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        image_name TEXT,
        image_data BLOB,
        image_format TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_instrument (
        instrument_id TEXT PRIMARY KEY,
        instrument_name TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_inst_setting (
        id INTEGER PRIMARY KEY,
        instrument_id TEXT NOT NULL REFERENCES flr_instrument(instrument_id),
        setting_name TEXT NOT NULL,
        setting_value TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_fret_analysis (
        analysis_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        sample_id TEXT REFERENCES flr_sample(sample_id),
        type TEXT,
        method TEXT,
        sample_probe_id_1 INTEGER,
        sample_probe_id_2 INTEGER,
        forster_radius_id INTEGER,
        dataset_list_id INTEGER,
        external_file_id INTEGER,
        software_id INTEGER,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_fret_calibration_parameters (
        id INTEGER PRIMARY KEY,
        analysis_id TEXT NOT NULL REFERENCES flr_fret_analysis(analysis_id),
        phi_acceptor REAL,
        alpha REAL,
        gamma REAL,
        delta REAL,
        beta REAL,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_fret_forster_radius (
        id INTEGER PRIMARY KEY,
        forster_radius_id TEXT UNIQUE,
        sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id),
        donor_probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        acceptor_probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        forster_radius REAL NOT NULL,
        reduced_forster_radius REAL,
        kappa_squared REAL DEFAULT 0.666667,
        index_of_refraction REAL DEFAULT 1.4,
        overlap_integral REAL,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (sample_id, donor_probe_id, acceptor_probe_id)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_fret_distance_restraint (
        id INTEGER PRIMARY KEY,
        analysis_id TEXT NOT NULL REFERENCES flr_fret_analysis(analysis_id),
        probe_id_1 INTEGER NOT NULL REFERENCES probes(probe_id),
        probe_id_2 INTEGER NOT NULL REFERENCES probes(probe_id),
        distance REAL,
        distance_error_plus REAL,
        distance_error_minus REAL,
        distance_type TEXT,
        state_id INTEGER,
        population_fraction REAL,
        peak_assignment_id INTEGER,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_entity_assembly (
        assembly_id TEXT PRIMARY KEY,
        description TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_exp_condition (
        condition_id TEXT PRIMARY KEY,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS ihm_chemical_component_descriptor (
        id INTEGER PRIMARY KEY,
        chemical_name TEXT,
        common_name TEXT,
        auth_name TEXT,
        smiles TEXT,
        smiles_canonical TEXT,
        inchi TEXT,
        inchi_key TEXT,
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS ihm_dataset_list (
        id INTEGER PRIMARY KEY,
        data_type TEXT,
        details TEXT,
        database_hosted TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS ihm_external_files (
        id INTEGER PRIMARY KEY,
        reference_id TEXT,
        file_path TEXT,
        file_format TEXT,
        content_type TEXT,
        file_size_bytes INTEGER,
        md5 TEXT,
        uuid TEXT,
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS analysis_metadata (
        analysis_id TEXT,
        key TEXT,
        value TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (analysis_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_photon_stream (
        stream_id TEXT PRIMARY KEY,
        analysis_id TEXT,
        external_file_id INTEGER REFERENCES ihm_external_files(id),
        detector_id TEXT,
        description TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS analysis_data (
        id INTEGER PRIMARY KEY,
        analysis_id TEXT NOT NULL REFERENCES flr_fret_analysis(analysis_id),
        data_type TEXT NOT NULL,
        data_name TEXT,
        x_values BLOB NOT NULL,
        y_values BLOB NOT NULL,
        x_unit TEXT,
        y_unit TEXT,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (analysis_id, data_type, data_name)
    )""",
    # Legacy fdb_setup (preserve as source for v17 backfill; canonical target is mfdb_setup)
    # Canonical FDB Tables (v17 target architecture — preserved for migration backfill compat)
    # Canonical MFDB Tables (v18 target architecture)
    """CREATE TABLE IF NOT EXISTS mfdb_object (
        object_uuid TEXT PRIMARY KEY,
        content_md5 TEXT NOT NULL UNIQUE,
        original_filename TEXT,
        size_bytes INTEGER,
        mime_type TEXT,
        storage_path TEXT NOT NULL,
        refcount INTEGER DEFAULT 1,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        created_by_user_uuid TEXT REFERENCES flr_sample_users(user_uuid)
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_artifact (
        artifact_id TEXT PRIMARY KEY,
        artifact_kind TEXT NOT NULL,
        data_format TEXT,
        experiment_id TEXT REFERENCES flr_experiment(experiment_id) ON DELETE SET NULL,
        storage_mode TEXT NOT NULL,
        file_path TEXT,
        url TEXT,
        folder_path TEXT,
        mime_type TEXT,
        size_bytes INTEGER,
        checksum TEXT,
        checksum_algorithm TEXT DEFAULT 'sha256',
        row_count INTEGER,
        validation_status TEXT DEFAULT 'unvalidated',
        validation_message TEXT,
        metadata_json TEXT,
        data_json TEXT,
        data_blob BLOB,
        object_uuid TEXT REFERENCES mfdb_object(object_uuid),
        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
        is_public INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_operation (
        operation_id TEXT PRIMARY KEY,
        operation_type TEXT NOT NULL,
        experiment_id TEXT REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        setup_id TEXT REFERENCES mfdb_setup(setup_id) ON DELETE SET NULL,
        settings_json TEXT,
        settings_hash TEXT,
        operator_user_id TEXT,
        software_package TEXT,
        software_module TEXT,
        software_version TEXT,
        runtime_environment_json TEXT,
        started_at TEXT,
        ended_at TEXT,
        status TEXT DEFAULT 'pending',
        error_message TEXT,
        traceback_summary TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_operation_artifact (
        operation_id TEXT NOT NULL REFERENCES mfdb_operation(operation_id) ON DELETE CASCADE,
        artifact_id TEXT NOT NULL REFERENCES mfdb_artifact(artifact_id) ON DELETE CASCADE,
        direction TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'generic',
        ordinal INTEGER DEFAULT 0,
        checksum_snapshot TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        PRIMARY KEY (operation_id, artifact_id, direction, role)
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_edge (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_node_type TEXT NOT NULL,
        source_node_id TEXT NOT NULL,
        target_node_type TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        operation_id TEXT REFERENCES mfdb_operation(operation_id) ON DELETE SET NULL,
        settings_hash TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        software_version TEXT,
        checksum_snapshot_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_parameter (
        parameter_id INTEGER PRIMARY KEY AUTOINCREMENT,
        parameter_uuid TEXT UNIQUE NOT NULL,
        operation_id TEXT REFERENCES mfdb_operation(operation_id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        value REAL,
        standard_error REAL,
        confidence_interval_low REAL,
        confidence_interval_high REAL,
        initial_value REAL,
        lower_bound REAL,
        upper_bound REAL,
        bounds_on INTEGER DEFAULT 0,
        units TEXT,
        parameter_type TEXT NOT NULL DEFAULT 'free',
        role TEXT,
        expression TEXT,
        prior_json TEXT,
        mapping_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_setup (
        setup_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version INTEGER DEFAULT 1,
        instrument_id TEXT REFERENCES flr_instrument(instrument_id),
        description TEXT,
        configuration_json TEXT,
        detectors_json TEXT,
        timing_calibration_json TEXT,
        irf_definition_json TEXT,
        dark_count_json TEXT,
        timing_resolution_json TEXT,
        macro_time_resolution REAL,
        micro_time_resolution REAL,
        micro_time_binning INTEGER,
        n_bins INTEGER DEFAULT 2,
        n_casc INTEGER DEFAULT 25,
        make_fine INTEGER DEFAULT 1,
        burst_defaults_json TEXT,
        fcs_calibration_json TEXT,
        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
        is_public INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    # Generated from the dictionary by _get_dict_ddl during module init
    _get_dict_ddl("mfdb_setup_detector_channel"),
    _get_dict_ddl("mfdb_setup_pie_window"),
    _get_dict_ddl("mfdb_setup_fcs_pair"),
    _get_dict_ddl("mfdb_setup_calibration"),
    _get_dict_ddl("mfdb_microtime_shift"),
    _get_dict_ddl("mfdb_artifact_owner"),
    """CREATE TABLE IF NOT EXISTS mfdb_audit_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        action TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        operator_user_id TEXT,
        details_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    # New MFDB Tables (v18)
    """CREATE TABLE IF NOT EXISTS mfdb_schema_version (version INTEGER)""",
    """CREATE TABLE IF NOT EXISTS mfdb_vocabulary (
        field_name TEXT NOT NULL,
        value TEXT NOT NULL,
        display_name TEXT,
        description TEXT,
        is_builtin INTEGER DEFAULT 0,
        is_active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        PRIMARY KEY (field_name, value)
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_branch (
        branch_uuid TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        parent_branch_uuid TEXT REFERENCES mfdb_branch(branch_uuid),
        head_operation_id TEXT REFERENCES mfdb_operation(operation_id),
        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    # Auth tables (v25)
    """CREATE TABLE IF NOT EXISTS mfdb_group (
        group_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        description TEXT,
        is_builtin INTEGER DEFAULT 0,
        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_group_member (
        group_id TEXT NOT NULL REFERENCES mfdb_group(group_id),
        user_id TEXT NOT NULL REFERENCES flr_sample_users(user_id),
        role TEXT DEFAULT 'member',
        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (group_id, user_id)
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_object_acl (
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        owner_user_id TEXT NOT NULL REFERENCES flr_sample_users(user_id),
        owner_group_id TEXT REFERENCES mfdb_group(group_id),
        mode INTEGER NOT NULL DEFAULT 448,
        inherits_from_type TEXT,
        inherits_from_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        UNIQUE (object_type, object_id)
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_acl_entry (
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        subject_type TEXT NOT NULL CHECK(subject_type IN ('user', 'group')),
        subject_id TEXT NOT NULL,
        effect TEXT NOT NULL CHECK(effect IN ('allow', 'deny')),
        permissions INTEGER NOT NULL,
        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_session (
        session_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES flr_sample_users(user_id),
        token_hash TEXT NOT NULL UNIQUE,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        expires_at TEXT NOT NULL,
        last_used_at TEXT,
        revoked_at TEXT,
        client_host TEXT,
        client_name TEXT,
        client_metadata_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_auth_attempt (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        client_host TEXT,
        success INTEGER NOT NULL DEFAULT 0,
        reason TEXT,
        attempted_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
]

# Canonical tables with CHECK constraints for production use.
# Used in the fresh-DB path only; the migration path uses
# CREATE_TABLES_SQL (without CHECK constraints) to avoid rejecting
# legacy data during backfill.
_CANONICAL_CHECK_SQL = [
    """CREATE TABLE IF NOT EXISTS mfdb_object (
        object_uuid TEXT PRIMARY KEY,
        content_md5 TEXT NOT NULL UNIQUE,
        original_filename TEXT,
        size_bytes INTEGER,
        mime_type TEXT,
        storage_path TEXT NOT NULL,
        refcount INTEGER DEFAULT 1,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        created_by_user_uuid TEXT REFERENCES flr_sample_users(user_uuid)
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_artifact (
        artifact_id TEXT PRIMARY KEY,
        artifact_kind TEXT NOT NULL,
        data_format TEXT,
        experiment_id TEXT REFERENCES flr_experiment(experiment_id) ON DELETE SET NULL,
        storage_mode TEXT NOT NULL
            CHECK (storage_mode IN ('local_file','local_directory','url','managed_archive',
                   'embedded_json','embedded_blob','local','remote','embedded','folder')),
        file_path TEXT,
        url TEXT,
        folder_path TEXT,
        mime_type TEXT,
        size_bytes INTEGER,
        checksum TEXT,
        checksum_algorithm TEXT DEFAULT 'sha256',
        row_count INTEGER,
        validation_status TEXT DEFAULT 'unvalidated'
            CHECK (validation_status IN ('unvalidated','valid','invalid','warning')),
        validation_message TEXT,
        metadata_json TEXT,
        data_json TEXT,
        data_blob BLOB,
        object_uuid TEXT REFERENCES mfdb_object(object_uuid),
        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
        is_public INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_operation (
        operation_id TEXT PRIMARY KEY,
        operation_type TEXT NOT NULL,
        experiment_id TEXT REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        setup_id TEXT REFERENCES mfdb_setup(setup_id) ON DELETE SET NULL,
        settings_json TEXT,
        settings_hash TEXT,
        operator_user_id TEXT,
        software_package TEXT,
        software_module TEXT,
        software_version TEXT,
        runtime_environment_json TEXT,
        started_at TEXT,
        ended_at TEXT,
        status TEXT DEFAULT 'pending'
            CHECK (status IN ('pending','running','succeeded','failed','cancelled','success','converged')),
        error_message TEXT,
        traceback_summary TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_operation_artifact (
        operation_id TEXT NOT NULL REFERENCES mfdb_operation(operation_id) ON DELETE CASCADE,
        artifact_id TEXT NOT NULL REFERENCES mfdb_artifact(artifact_id) ON DELETE CASCADE,
        direction TEXT NOT NULL
            CHECK (direction IN ('input','output')),
        role TEXT NOT NULL DEFAULT 'generic',
        ordinal INTEGER DEFAULT 0,
        checksum_snapshot TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        PRIMARY KEY (operation_id, artifact_id, direction, role)
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_edge (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_node_type TEXT NOT NULL,
        source_node_id TEXT NOT NULL,
        target_node_type TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL
            CHECK (relationship_type NOT IN ('input_to','produced')),
        operation_id TEXT REFERENCES mfdb_operation(operation_id) ON DELETE SET NULL,
        settings_hash TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        software_version TEXT,
        checksum_snapshot_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_parameter (
        parameter_id INTEGER PRIMARY KEY AUTOINCREMENT,
        parameter_uuid TEXT UNIQUE NOT NULL,
        operation_id TEXT REFERENCES mfdb_operation(operation_id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        value REAL,
        standard_error REAL,
        confidence_interval_low REAL,
        confidence_interval_high REAL,
        initial_value REAL,
        lower_bound REAL,
        upper_bound REAL,
        bounds_on INTEGER DEFAULT 0,
        units TEXT,
        parameter_type TEXT NOT NULL DEFAULT 'free',
        role TEXT,
        expression TEXT,
        prior_json TEXT,
        mapping_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
]

# ---------------------------------------------------------------------------
# Extension tables managed by reconcile_schema
# ---------------------------------------------------------------------------


# Fresh-DB tables — same as CREATE_TABLES_SQL but with CHECK-constrained
# canonical table definitions for the mfdb_* canonical tables.
_CANONICAL_TABLE_MAP: dict[str, str] = {}

# Build CHECK-constrained entries — match on mfdb_* prefix
for csql in _CANONICAL_CHECK_SQL:
    for mfdb_name in [
        "mfdb_artifact", "mfdb_operation",
        "mfdb_operation_artifact", "mfdb_edge", "mfdb_parameter",
    ]:
        prefix = f"CREATE TABLE IF NOT EXISTS {mfdb_name} "
        if prefix in csql:
            _CANONICAL_TABLE_MAP[mfdb_name] = csql
            break

FRESH_DB_TABLES_SQL = []
for sql in CREATE_TABLES_SQL:
    for mfdb_name, replacement in _CANONICAL_TABLE_MAP.items():
        prefix = f"CREATE TABLE IF NOT EXISTS {mfdb_name} "
        if prefix in sql:
            sql = replacement
            break
    FRESH_DB_TABLES_SQL.append(sql)

CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_probes_type ON probes (type_id)",
    "CREATE INDEX IF NOT EXISTS idx_probes_cat ON probes (category)",
    "CREATE INDEX IF NOT EXISTS idx_op_probe ON optical_properties (probe_id)",
    "CREATE INDEX IF NOT EXISTS idx_spectra_probe ON spectra (probe_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_sample ON flr_experiment (sample_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_type ON flr_experiment (type_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_data_experiment ON flr_experiment_data (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_key_value_experiment ON flr_experiment_key_value (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_setup ON flr_experiment (setup_definition_id)",
    # Canonical MFDB Indices
    "CREATE INDEX IF NOT EXISTS idx_mfdb_artifact_kind ON mfdb_artifact (artifact_kind)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_artifact_data_format ON mfdb_artifact (data_format)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_artifact_experiment ON mfdb_artifact (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_operation_type ON mfdb_operation (operation_type)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_operation_experiment ON mfdb_operation (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_operation_setup ON mfdb_operation (setup_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_operation_status ON mfdb_operation (status)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_op_art_op ON mfdb_operation_artifact (operation_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_op_art_art ON mfdb_operation_artifact (artifact_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_edge_source ON mfdb_edge (source_node_type, source_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_edge_target ON mfdb_edge (target_node_type, target_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_edge_operation ON mfdb_edge (operation_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_parameter_operation ON mfdb_parameter (operation_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_parameter_uuid ON mfdb_parameter (parameter_uuid)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_setup_instrument ON mfdb_setup (instrument_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_setup_detector_channel_setup ON mfdb_setup_detector_channel (setup_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_setup_pie_window_setup ON mfdb_setup_pie_window (setup_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_setup_fcs_pair_setup ON mfdb_setup_fcs_pair (setup_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_artifact_owner_user ON mfdb_artifact_owner (user_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_mfdb_artifact_owner_uniq ON mfdb_artifact_owner (artifact_id, user_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_setup_calibration_setup ON mfdb_setup_calibration (setup_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_microtime_shift_operation ON mfdb_microtime_shift (operation_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_audit_log_target ON mfdb_audit_log (target_type, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_audit_log_timestamp ON mfdb_audit_log (timestamp)",
    # (mfdb_sample/mfdb_experiment indices removed in Phase 3 — tables are duplicates of flr_*)
    "CREATE INDEX IF NOT EXISTS idx_mfdb_vocabulary_field ON mfdb_vocabulary (field_name)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_vocabulary_active ON mfdb_vocabulary (is_active)",
    # Lifecycle indices (v23)
    "CREATE INDEX IF NOT EXISTS idx_probes_created_at ON probes (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_probes_updated_at ON probes (updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_probes_deleted_at ON probes (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_entities_deleted_at ON entities (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_flr_sample_deleted_at ON flr_sample (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_flr_sample_users_deleted_at ON flr_sample_users (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_flr_sample_devices_deleted_at ON flr_sample_devices (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_flr_experiment_deleted_at ON flr_experiment (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_flr_experiment_data_deleted_at ON flr_experiment_data (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_flr_fret_analysis_deleted_at ON flr_fret_analysis (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_analysis_metadata_deleted_at ON analysis_metadata (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_artifact_deleted_at ON mfdb_artifact (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_operation_deleted_at ON mfdb_operation (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_operation_artifact_deleted_at ON mfdb_operation_artifact (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_edge_deleted_at ON mfdb_edge (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_parameter_deleted_at ON mfdb_parameter (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_setup_deleted_at ON mfdb_setup (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_audit_log_deleted_at ON mfdb_audit_log (deleted_at)",
    # (mfdb_sample/mfdb_experiment lifecycle indices removed in Phase 3)
    "CREATE INDEX IF NOT EXISTS idx_mfdb_vocabulary_deleted_at ON mfdb_vocabulary (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_branch_deleted_at ON mfdb_branch (deleted_at)",
    # Auth indices (v25)
    "CREATE INDEX IF NOT EXISTS idx_mfdb_group_member_group ON mfdb_group_member (group_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_group_member_user ON mfdb_group_member (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_object_acl_owner ON mfdb_object_acl (owner_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_object_acl_owner_group ON mfdb_object_acl (owner_group_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_object_acl_object ON mfdb_object_acl (object_type, object_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_acl_entry_object ON mfdb_acl_entry (object_type, object_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_acl_entry_subject ON mfdb_acl_entry (subject_type, subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_session_token_hash ON mfdb_session (token_hash)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_session_expires ON mfdb_session (expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_session_user ON mfdb_session (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_auth_attempt_user ON mfdb_auth_attempt (user_id, attempted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_auth_attempt_host ON mfdb_auth_attempt (client_host, attempted_at)",
    # Object store indices (v27)
    "CREATE INDEX IF NOT EXISTS idx_mfdb_object_md5 ON mfdb_object (content_md5)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_object_filename ON mfdb_object (original_filename)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_artifact_object_uuid ON mfdb_artifact (object_uuid)",
]

# Fresh-DB indices — same as CREATE_INDICES_SQL but without legacy fdb_* indices.
FRESH_DB_INDICES_SQL = [
    sql for sql in CREATE_INDICES_SQL
    if " ON fdb_" not in sql
]


def _safe_table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add *column* to *table* when it is missing."""
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if table not in existing:
            return
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.OperationalError:
        return
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _ensure_mfdb_setup_columns(conn: sqlite3.Connection) -> None:
    """Ensure all typed columns added by v31-v33 migrations exist on
    ``mfdb_setup``.

    This is a defensive repair for databases that were migrated to
    v32/v33 but silently missed column additions due to the previous
    ``try/except OperationalError: pass`` pattern in the migration code.
    It is idempotent and safe to call on every connection open.
    """
    for col, col_type in [
        ("macro_time_resolution", "REAL"),
        ("micro_time_resolution", "REAL"),
        ("micro_time_binning", "INTEGER"),
        ("created_by_user_id", "TEXT REFERENCES flr_sample_users(user_id)"),
        ("is_public", "INTEGER DEFAULT 0"),
        ("n_bins", "INTEGER DEFAULT 2"),
        ("n_casc", "INTEGER DEFAULT 25"),
        ("make_fine", "INTEGER DEFAULT 1"),
    ]:
        _ensure_column(conn, "mfdb_setup", col, col_type)


def _fix_operation_artifact_pk(conn: sqlite3.Connection) -> None:
    """Recreate ``mfdb_operation_artifact`` with a 4-column composite PK.

    Older databases were created with
    ``PRIMARY KEY (operation_id, artifact_id, direction)`` but the code now
    uses ``ON CONFLICT(operation_id, artifact_id, direction, role)`` which
    requires *role* in the PK.  SQLite does not support ``ALTER TABLE … ADD
    PRIMARY KEY`` so we recreate the table.
    """
    cur = conn.cursor()
    pk_cols = {
        r[1] for r in cur.execute("PRAGMA table_info(mfdb_operation_artifact)").fetchall()
    }
    # If 'role' is already in the PK we have nothing to do
    if not pk_cols:
        return  # table doesn't exist yet, nothing to fix
    # Check if 'role' column is part of the PK already
    role_in_pk = any(
        r[1] == "role" and r[5]
        for r in cur.execute("PRAGMA table_info(mfdb_operation_artifact)").fetchall()
    )
    if role_in_pk:
        return
    logger.info("Migrating mfdb_operation_artifact: adding 'role' to composite PK")
    cur.execute("BEGIN")
    try:
        cur.execute("CREATE TABLE mfdb_operation_artifact_new ("
                     "operation_id TEXT NOT NULL REFERENCES mfdb_operation(operation_id) ON DELETE CASCADE, "
                     "artifact_id TEXT NOT NULL REFERENCES mfdb_artifact(artifact_id) ON DELETE CASCADE, "
                     "direction TEXT NOT NULL, "
                     "role TEXT NOT NULL DEFAULT 'generic', "
                     "ordinal INTEGER DEFAULT 0, "
                     "checksum_snapshot TEXT, "
                     "metadata_json TEXT, "
                     "created_at TEXT, updated_at TEXT, deleted_at TEXT, "
                     "PRIMARY KEY (operation_id, artifact_id, direction, role))")
        cur.execute("INSERT INTO mfdb_operation_artifact_new "
                     "(operation_id, artifact_id, direction, role, ordinal, "
                     "checksum_snapshot, metadata_json, created_at, updated_at, deleted_at) "
                     "SELECT operation_id, artifact_id, direction, "
                     "COALESCE(role, 'generic'), ordinal, "
                     "checksum_snapshot, metadata_json, created_at, updated_at, deleted_at "
                     "FROM mfdb_operation_artifact")
        cur.execute("DROP TABLE mfdb_operation_artifact")
        cur.execute("ALTER TABLE mfdb_operation_artifact_new RENAME TO mfdb_operation_artifact")
        conn.commit()
        logger.info("mfdb_operation_artifact PK migration complete")
    except Exception:
        conn.rollback()
        logger.warning("mfdb_operation_artifact PK migration skipped (table may already be correct)")


def _fix_flr_fret_forster_radius_sample_id(conn: sqlite3.Connection) -> bool:
    """Recreate ``flr_fret_forster_radius`` with sample_id column, forster_radius_id column, and scoped uniqueness.

    R14 added a ``sample_id`` column to scope FRET pairs to specific samples,
    but existing v28 databases were created without this column, causing
    ``sqlite3.OperationalError: no such column: sample_id`` when the updated
    code tries to insert/query it.

    R15-6 added the ``forster_radius_id`` column to store deterministic IDs.

    SQLite does not support adding columns with NOT NULL constraints or
    changing UNIQUE constraints, so we recreate the table.

    Returns
    -------
    bool
        True if migration succeeded, False otherwise.
    """
    cur = conn.cursor()

    # Initialize counters for logging
    migrated_count = 0
    ambiguous_count = 0

    # Check if the table exists
    table_info = cur.execute("PRAGMA table_info(flr_fret_forster_radius)").fetchall()
    if not table_info:
        # Table doesn't exist, nothing to migrate
        return True

    # Check if both sample_id and forster_radius_id columns exist
    column_names = {r[1] for r in table_info}
    has_sample_id = "sample_id" in column_names
    has_forster_radius_id = "forster_radius_id" in column_names

    if has_sample_id and has_forster_radius_id:
        # Both columns exist, migration already done
        return True

    logger.info("Migrating flr_fret_forster_radius: adding sample_id, forster_radius_id columns and scoped UNIQUE constraint")
    cur.execute("BEGIN")
    try:
        # Create new table with correct schema
        cur.execute("""CREATE TABLE flr_fret_forster_radius_new (
            id INTEGER PRIMARY KEY,
            forster_radius_id TEXT UNIQUE,
            sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id),
            donor_probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
            acceptor_probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
            forster_radius REAL NOT NULL,
            reduced_forster_radius REAL,
            kappa_squared REAL DEFAULT 0.666667,
            index_of_refraction REAL DEFAULT 1.4,
            overlap_integral REAL,
            details TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            deleted_at TEXT,
            UNIQUE (sample_id, donor_probe_id, acceptor_probe_id)
        )""")

        # Try to copy existing data if possible
        # If old table has sample_id, we can migrate directly
        if has_sample_id:
            old_rows = cur.execute(
                "SELECT id, sample_id, donor_probe_id, acceptor_probe_id, "
                "forster_radius, reduced_forster_radius, kappa_squared, "
                "index_of_refraction, overlap_integral, details, created_at, "
                "updated_at, deleted_at FROM flr_fret_forster_radius"
            ).fetchall()

            for old_row in old_rows:
                # Generate a deterministic forster_radius_id
                forster_radius_id = f"{old_row[1]}_forster_{old_row[0]}"  # sample_id_forster_id
                cur.execute(
                    "INSERT INTO flr_fret_forster_radius_new "
                    "(id, forster_radius_id, sample_id, donor_probe_id, acceptor_probe_id, "
                    "forster_radius, reduced_forster_radius, kappa_squared, "
                    "index_of_refraction, overlap_integral, details, created_at, "
                    "updated_at, deleted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (old_row[0], forster_radius_id, old_row[1], old_row[2], old_row[3],
                     old_row[4], old_row[5], old_row[6], old_row[7], old_row[8],
                     old_row[9], old_row[10], old_row[11], old_row[12])
                )
        else:
            # Old table doesn't have sample_id - try to backfill from flr_sample_probe
            # For each FRET pair row, find the common sample_id that uses both probes
            old_rows = cur.execute(
                "SELECT id, donor_probe_id, acceptor_probe_id, "
                "forster_radius, reduced_forster_radius, kappa_squared, "
                "index_of_refraction, overlap_integral, details, created_at, "
                "updated_at, deleted_at FROM flr_fret_forster_radius"
            ).fetchall()

            for old_row in old_rows:
                donor_probe_id = old_row[1]
                acceptor_probe_id = old_row[2]

                # Find all samples that use the donor probe
                donor_samples = cur.execute(
                    "SELECT DISTINCT sample_id FROM flr_sample_probe "
                    "WHERE probe_id = ? AND deleted_at IS NULL",
                    (donor_probe_id,)
                ).fetchall()

                # Find all samples that use the acceptor probe
                acceptor_samples = cur.execute(
                    "SELECT DISTINCT sample_id FROM flr_sample_probe "
                    "WHERE probe_id = ? AND deleted_at IS NULL",
                    (acceptor_probe_id,)
                ).fetchall()

                donor_sample_ids = {r[0] for r in donor_samples}
                acceptor_sample_ids = {r[0] for r in acceptor_samples}

                # Find common samples
                common_samples = donor_sample_ids & acceptor_sample_ids

                if len(common_samples) == 1:
                    # Unambiguous: both probes belong to exactly one common sample
                    sample_id = list(common_samples)[0]
                    forster_radius_id = f"{sample_id}_forster_{old_row[0]}"
                    cur.execute(
                        "INSERT INTO flr_fret_forster_radius_new "
                        "(id, forster_radius_id, sample_id, donor_probe_id, acceptor_probe_id, "
                        "forster_radius, reduced_forster_radius, kappa_squared, "
                        "index_of_refraction, overlap_integral, details, created_at, "
                        "updated_at, deleted_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (old_row[0], forster_radius_id, sample_id, old_row[1], old_row[2],
                         old_row[3], old_row[4], old_row[5], old_row[6], old_row[7],
                         old_row[8], old_row[9], old_row[10], old_row[11])
                    )
                    migrated_count += 1
                elif len(common_samples) > 1:
                    # Ambiguous: probes belong to multiple common samples
                    # Log for manual repair
                    ambiguous_count += 1
                    logger.warning(
                        f"Ambiguous FRET pair (id={old_row[0]}): donor_probe_id={donor_probe_id} "
                        f"and acceptor_probe_id={acceptor_probe_id} found in multiple samples: "
                        f"{common_samples}. This row was not migrated automatically."
                    )
                else:
                    # No common sample found - probes not linked through flr_sample_probe
                    # Try to find sample from flr_fret_analysis if it exists
                    logger.warning(
                        f"FRET pair (id={old_row[0]}): donor_probe_id={donor_probe_id} "
                        f"and acceptor_probe_id={acceptor_probe_id} not linked to any common sample "
                        f"via flr_sample_probe. This row was not migrated automatically."
                    )

            if migrated_count > 0:
                logger.info(f"Migrated {migrated_count} unambiguous FRET pair(s)")
            if ambiguous_count > 0:
                logger.warning(f"{ambiguous_count} FRET pair(s) were ambiguous and require manual repair")

        # Drop old table
        cur.execute("DROP TABLE flr_fret_forster_radius")

        # Rename new table
        cur.execute("ALTER TABLE flr_fret_forster_radius_new RENAME TO flr_fret_forster_radius")

        conn.commit()
        if has_sample_id:
            logger.info("flr_fret_forster_radius migration complete - existing data migrated")
        else:
            if migrated_count > 0 and ambiguous_count == 0:
                logger.info(f"flr_fret_forster_radius migration complete - backfilled {migrated_count} unambiguous FRET pair(s)")
            elif migrated_count > 0 and ambiguous_count > 0:
                logger.info(f"flr_fret_forster_radius migration complete - backfilled {migrated_count} unambiguous FRET pair(s), {ambiguous_count} ambiguous pair(s) require manual repair")
            else:
                logger.info("flr_fret_forster_radius migration complete - no legacy FRET pair data to migrate")
        return True
    except Exception as e:
        conn.rollback()
        logger.warning("flr_fret_forster_radius migration failed: %s", e)
        return False


def _ensure_lifecycle_columns(conn: sqlite3.Connection, now: str | None = None) -> None:
    """Repair lifecycle columns on legacy MFDB tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite connection to repair.
    now : str or None, optional
        Timestamp used for backfilling missing timestamps.
    """
    now = now or _utc_now()
    lifecycle_table_config: list[tuple[str, bool, bool]] = [
        ("mfdb_artifact", True, True),
        ("mfdb_operation", True, True),
        ("mfdb_parameter", True, True),
        ("mfdb_setup", True, True),
        ("mfdb_setup_detector_channel", True, True),
        ("mfdb_setup_pie_window", True, True),
        ("mfdb_setup_fcs_pair", True, True),
        ("mfdb_setup_calibration", True, True),
        ("mfdb_microtime_shift", True, True),
        ("mfdb_artifact_owner", True, True),
        ("mfdb_branch", True, True),
        ("mfdb_audit_log", True, False),
        ("probes", False, False),
        ("entities", False, False),
        ("entity_poly_seq", False, False),
        ("flr_sample", False, False),
        ("flr_sample_condition", False, False),
        ("flr_sample_users", False, False),
        ("flr_sample_devices", False, False),
        ("flr_experiment_type", False, False),
        ("flr_experiment", False, False),
        ("flr_experiment_key_value", False, False),
        ("flr_experiment_data", False, False),
        ("flr_sample_key_value", False, False),
        ("flr_sample_probe", False, False),
        ("flr_poly_probe_position", False, False),
        ("optical_properties", False, False),
        ("spectra", False, False),
        ("images", False, False),
        ("flr_instrument", False, False),
        ("flr_inst_setting", False, False),
        ("flr_fret_analysis", False, False),
        ("flr_fret_calibration_parameters", False, False),
        ("flr_fret_forster_radius", False, False),
        ("flr_fret_distance_restraint", False, False),
        ("flr_entity_assembly", False, False),
        ("flr_exp_condition", False, False),
        ("analysis_metadata", False, False),
        ("flr_photon_stream", False, False),
        ("analysis_data", False, False),
        ("mfdb_operation_artifact", False, False),
        ("mfdb_edge", False, False),
        ("mfdb_vocabulary", False, False),
        ("mfdb_group", False, False),
        ("mfdb_group_member", False, False),
        ("mfdb_object_acl", False, False),
        ("mfdb_acl_entry", False, False),
        ("mfdb_session", False, False),
        ("mfdb_auth_attempt", False, False),
    ]

    with conn:
        for table, has_created, has_updated in lifecycle_table_config:
            try:
                _ensure_column(conn, table, "created_at", "TEXT")
                _ensure_column(conn, table, "updated_at", "TEXT")
                _ensure_column(conn, table, "deleted_at", "TEXT")
                conn.execute(f"UPDATE {table} SET created_at = ? WHERE created_at IS NULL", (now,))
                if not has_updated:
                    conn.execute(f"UPDATE {table} SET updated_at = ? WHERE updated_at IS NULL", (now,))
            except sqlite3.OperationalError:
                pass


def get_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT version FROM mfdb_schema_version").fetchone()
        if row is not None:
            return row[0]
    except sqlite3.OperationalError:
        pass
    try:
        row = conn.execute("SELECT version FROM _schema_version").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def set_schema_version(conn: sqlite3.Connection, version: int):
    try:
        conn.execute("DELETE FROM mfdb_schema_version")
        conn.execute("INSERT INTO mfdb_schema_version (version) VALUES (?)", (version,))
    except sqlite3.OperationalError:
        pass
    conn.execute("DELETE FROM _schema_version")
    conn.execute("INSERT INTO _schema_version (version) VALUES (?)", (version,))


def _hash_admin_password() -> str:
    """Hash the default admin password 'admin' (PBKDF2-SHA256)."""
    import hashlib
    salt = "a1b2c3d4e5f6a7b8"  # fixed salt for reproducibility
    iterations = 100000
    dk = hashlib.pbkdf2_hmac(
        "sha256", b"admin", salt.encode("utf-8"), iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"


def bootstrap_default_user(conn: sqlite3.Connection) -> None:
    """Bootstrap the default admin and guest users in flr_sample_users.

    Parameters
    ----------
    conn : sqlite3.Connection
        The database connection.
    """
    import uuid
    main_uuid = "00000000-0000-0000-0000-000000000000"
    with conn:
        row_main = conn.execute("SELECT branch_uuid FROM mfdb_branch WHERE name = 'main'").fetchone()
        if not row_main:
            conn.execute(
                "INSERT INTO mfdb_branch (branch_uuid, name, description) VALUES (?, 'main', 'Default main branch')",
                (main_uuid,)
            )
        else:
            main_uuid = row_main[0]

        row = conn.execute("SELECT user_uuid, active_branch_uuid FROM flr_sample_users WHERE user_id = 'user_default'").fetchone()
        if not row:
            user_count = conn.execute("SELECT COUNT(*) FROM flr_sample_users").fetchone()[0]
            if user_count:
                conn.execute(
                    "UPDATE flr_sample_users SET is_admin = 1, password_hash = ? WHERE user_id = 'user_default'",
                    (_hash_admin_password(),)
                )
                return
            conn.execute(
                "INSERT INTO flr_sample_users (user_id, user_uuid, display_name, active_branch_uuid, is_admin, password_hash) VALUES ('user_default', ?, 'Default User', ?, 1, ?)",
                (str(uuid.uuid4()), main_uuid, _hash_admin_password())
            )
        elif not row[0] or not row[1]:
            u_uuid, a_uuid = row
            if not u_uuid:
                conn.execute(
                    "UPDATE flr_sample_users SET user_uuid = ? WHERE user_id = 'user_default'",
                    (str(uuid.uuid4()),)
                )
            if not a_uuid:
                conn.execute(
                    "UPDATE flr_sample_users SET active_branch_uuid = ? WHERE user_id = 'user_default'",
                    (main_uuid,)
                )
        conn.execute(
            "UPDATE flr_sample_users SET is_admin = 1, password_hash = COALESCE(password_hash, ?) WHERE user_id = 'user_default'",
            (_hash_admin_password(),)
        )

        # Create guest user (passwordless, non-admin)
        guest_row = conn.execute(
            "SELECT user_uuid FROM flr_sample_users WHERE user_id = 'guest'"
        ).fetchone()
        if not guest_row:
            conn.execute(
                "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name, active_branch_uuid, is_admin, allow_passwordless_login) VALUES ('guest', ?, 'Guest User', ?, 0, 1)",
                (str(uuid.uuid4()), main_uuid)
            )
        else:
            conn.execute(
                "UPDATE flr_sample_users SET allow_passwordless_login = 1 WHERE user_id = 'guest' AND (allow_passwordless_login IS NULL OR allow_passwordless_login != 1)"
            )


def bootstrap_auth_groups(conn: sqlite3.Connection) -> None:
    """Bootstrap built-in auth groups (admins, users, public) and add existing users."""
    with conn:
        for group_id, display_name, description in [
            ("admins", "Administrators", "Built-in admin group — members have full access"),
            ("users", "Users", "Built-in users group — normal authenticated users"),
            ("public", "Public", "Conceptual anonymous/public group"),
        ]:
            conn.execute(
                """INSERT OR IGNORE INTO mfdb_group (group_id, display_name, description, is_builtin)
                   VALUES (?, ?, ?, 1)""",
                (group_id, display_name, description),
            )

        try:
            existing_users = conn.execute(
                "SELECT user_id FROM flr_sample_users WHERE deleted_at IS NULL"
            ).fetchall()
        except sqlite3.OperationalError:
            existing_users = conn.execute(
                "SELECT user_id FROM flr_sample_users"
            ).fetchall()

        for row in existing_users:
            uid = row["user_id"] if isinstance(row, dict) else row[0]
            conn.execute(
                """INSERT OR IGNORE INTO mfdb_group_member (group_id, user_id, role)
                   VALUES ('users', ?, 'member')""",
                (uid,),
            )

        try:
            admin_users = conn.execute(
                "SELECT user_id FROM flr_sample_users WHERE is_admin = 1 AND (deleted_at IS NULL OR deleted_at = '')"
            ).fetchall()
        except sqlite3.OperationalError:
            try:
                admin_users = conn.execute(
                    "SELECT user_id FROM flr_sample_users WHERE is_admin = 1"
                ).fetchall()
            except sqlite3.OperationalError:
                admin_users = []

        for row in admin_users:
            uid = row["user_id"] if isinstance(row, dict) else row[0]
            conn.execute(
                """INSERT OR IGNORE INTO mfdb_group_member (group_id, user_id, role)
                   VALUES ('admins', ?, 'member')""",
                (uid,),
            )


def bootstrap_vocabulary(conn: sqlite3.Connection) -> None:
    """Bootstrap built-in and legacy extensible vocabulary values in mfdb_vocabulary.

    All enumeration values are read from the bundled mmCIF/flrCIF dictionary,
    including MFDB extension categories defined in ``mfdb_flr_ext.dic``.
    """
    from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary

    dic = MmcifDictionary.load_bundled()

    # Map vocab field_name → dictionary item full_name
    _field_to_item: dict[str, str] = {
        "artifact_kind": "_mfdb_artifact.artifact_kind",
        "data_format": "_mfdb_artifact.data_format",
        "operation_type": "_mfdb_operation.operation_type",
        "parameter_type": "_mfdb_parameter.parameter_type",
        "relationship_type": "_mfdb_edge.relationship_type",
        "direction": "_mfdb_operation_artifact.direction",
        "status": "_mfdb_operation.status",
        "validation_status": "_mfdb_operation.validation_status",
        "storage_mode": "_mfdb_object.storage_mode",
        "lifecycle_status": "_mfdb_branch.lifecycle_status",
        "sample_type": "_flr_sample.sample_type",
        "entity_type": "_entity.type",
        "fluorophore_type": "_flr_sample_probe_details.fluorophore_type",
        "solvent_phase": "_flr_sample.solvent_phase",
        "probe_origin": "_flr_probe_list.probe_origin",
        "probe_link_type": "_flr_probe_list.probe_link_type",
        "reactive_probe_flag": "_flr_probe_list.reactive_probe_flag",
        "ambiguous_stoichiometry": "_flr_poly_probe_position.mutation_flag",
    }

    with conn:
        for field_name, item_path in _field_to_item.items():
            values = [v for v in dic.get_enumerations(item_path) if v not in {"#", ".", "?"}]
            for val in values:
                conn.execute(
                    """INSERT OR IGNORE INTO mfdb_vocabulary (
                        field_name, value, display_name, description, is_builtin, is_active
                    ) VALUES (?, ?, ?, ?, 1, 1)""",
                    (field_name, val, val, f"Built-in {field_name} value")
                )


def _edge_has_relationship_constraint(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='mfdb_edge'"
    ).fetchone()
    return bool(row and "relationship_type NOT IN ('input_to','produced')" in (row[0] or ""))


def _copy_non_operation_edges(conn: sqlite3.Connection, source_table: str, target_table: str) -> int:
    inserted = 0
    for row in conn.execute(f"SELECT * FROM {source_table}").fetchall():
        if row["relationship_type"] in ("input_to", "produced"):
            continue
        conn.execute(
            f"""INSERT INTO {target_table} (
                source_node_type, source_node_id, target_node_type, target_node_id,
                relationship_type, operation_id, settings_hash, timestamp, software_version,
                checksum_snapshot_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["source_node_type"], row["source_node_id"], row["target_node_type"],
                row["target_node_id"], row["relationship_type"], row["operation_id"],
                row["settings_hash"], row["timestamp"], row["software_version"],
                row["checksum_snapshot_json"], row["metadata_json"],
            ),
        )
        inserted += 1
    return inserted


def _convert_operation_edges(conn: sqlite3.Connection) -> dict[str, int | list[str]]:
    converted = 0
    skipped = 0
    warnings: list[str] = []
    for row in conn.execute(
        "SELECT * FROM mfdb_edge WHERE relationship_type IN ('input_to','produced')"
    ).fetchall():
        rel_type = row["relationship_type"]
        if rel_type == "input_to" and row["target_node_type"] in ("operation", "processing_run", "analysis_run"):
            conn.execute(
                """INSERT OR IGNORE INTO mfdb_operation_artifact (
                    operation_id, artifact_id, direction, role, metadata_json
                ) VALUES (?, ?, 'input', ?, ?)""",
                (
                    row["target_node_id"], row["source_node_id"],
                    row["source_node_type"], row["metadata_json"],
                ),
            )
            converted += 1
        elif rel_type == "produced" and row["source_node_type"] in ("operation", "processing_run", "analysis_run"):
            conn.execute(
                """INSERT OR IGNORE INTO mfdb_operation_artifact (
                    operation_id, artifact_id, direction, role, metadata_json
                ) VALUES (?, ?, 'output', ?, ?)""",
                (
                    row["source_node_id"], row["target_node_id"],
                    row["target_node_type"], row["metadata_json"],
                ),
            )
            converted += 1
        else:
            skipped += 1
            warnings.append(f"Could not convert mfdb_edge:{row['edge_id']} {rel_type}")
    conn.execute("DELETE FROM mfdb_edge WHERE relationship_type IN ('input_to','produced')")
    return {"converted": converted, "skipped": skipped, "warnings": warnings}


def _ensure_mfdb_edge_constraint(conn: sqlite3.Connection) -> dict[str, int | list[str]]:
    if _edge_has_relationship_constraint(conn):
        return {"converted": 0, "skipped": 0, "warnings": []}

    conversion = _convert_operation_edges(conn)
    for index_name in (
        "idx_mfdb_edge_source", "idx_mfdb_edge_target", "idx_mfdb_edge_operation"
    ):
        conn.execute(f"DROP INDEX IF EXISTS {index_name}")

    legacy_table = "__mfdb_edge_legacy"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute("ALTER TABLE mfdb_edge RENAME TO __mfdb_edge_legacy")
    conn.execute(_CANONICAL_TABLE_MAP["mfdb_edge"])
    inserted = _copy_non_operation_edges(conn, legacy_table, "mfdb_edge")
    conn.execute("DROP TABLE __mfdb_edge_legacy")
    for sql in CREATE_INDICES_SQL:
        if "idx_mfdb_edge" in sql:
            conn.execute(sql)
    conversion["inserted"] = inserted
    return conversion


MFDB_EDGE_VOCABULARY_TRIGGER_SQL = [
    """CREATE TRIGGER IF NOT EXISTS trg_mfdb_edge_relationship_type_insert
    BEFORE INSERT ON mfdb_edge
    FOR EACH ROW
    BEGIN
        SELECT CASE
            WHEN NEW.relationship_type IN ('input_to', 'produced') THEN
                RAISE(ABORT, 'mfdb_edge.relationship_type must not be input_to or produced')
            WHEN NOT EXISTS (
                SELECT 1 FROM mfdb_vocabulary
                WHERE field_name = 'relationship_type'
                  AND value = NEW.relationship_type
                  AND is_active = 1
            ) THEN
                RAISE(ABORT, 'mfdb_edge.relationship_type is not an active vocabulary value')
        END;
    END""",
    """CREATE TRIGGER IF NOT EXISTS trg_mfdb_edge_relationship_type_update
    BEFORE UPDATE OF relationship_type ON mfdb_edge
    FOR EACH ROW
    BEGIN
        SELECT CASE
            WHEN NEW.relationship_type IN ('input_to', 'produced') THEN
                RAISE(ABORT, 'mfdb_edge.relationship_type must not be input_to or produced')
            WHEN NOT EXISTS (
                SELECT 1 FROM mfdb_vocabulary
                WHERE field_name = 'relationship_type'
                  AND value = NEW.relationship_type
                  AND is_active = 1
            ) THEN
                RAISE(ABORT, 'mfdb_edge.relationship_type is not an active vocabulary value')
        END;
    END""",
]


def _ensure_mfdb_edge_vocabulary_triggers(conn: sqlite3.Connection) -> dict[str, int | list[str]]:
    invalid_rows = conn.execute(
        """SELECT edge_id, relationship_type
           FROM mfdb_edge
           WHERE relationship_type IN ('input_to', 'produced')
              OR NOT EXISTS (
                  SELECT 1 FROM mfdb_vocabulary
                  WHERE field_name = 'relationship_type'
                    AND value = mfdb_edge.relationship_type
                    AND is_active = 1
              )"""
    ).fetchall()
    if invalid_rows:
        details = ", ".join(
            f"edge {row['edge_id']}={row['relationship_type']}" for row in invalid_rows
        )
        raise ValueError(f"Invalid mfdb_edge.relationship_type values: {details}")
    for sql in MFDB_EDGE_VOCABULARY_TRIGGER_SQL:
        conn.execute(sql)
    return {"source": 0, "inserted": 0, "skipped": 0, "warnings": []}


FRESH_DB_SCHEMA_SQL = FRESH_DB_TABLES_SQL + FRESH_DB_INDICES_SQL + MFDB_EDGE_VOCABULARY_TRIGGER_SQL


def migrate_schema(conn: sqlite3.Connection) -> MigrationReport | None:
    """Execution of versioned migration logic, including canonical schema v18.

    Returns
    -------
    MigrationReport or None
        A report if migration was performed, or ``None`` if the schema was
        already at the current version.
    """
    report: MigrationReport | None = None
    # Run the legacy migration logic from the original schema
    with conn:
        cursor = conn.cursor()
        try:
            existing = {
                r[0]
                for r in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if not existing:
                for sql in FRESH_DB_SCHEMA_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, SCHEMA_VERSION)
                bootstrap_vocabulary(conn)
                bootstrap_default_user(conn)
                bootstrap_auth_groups(conn)
                from chisurf.core.mfdb.schema_from_dictionary import reconcile_schema
                from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
                reconcile_schema(conn, MmcifDictionary.load_bundled())
                from chisurf.core.mfdb.operation_parameters import (
                    bootstrap_operation_parameter_defs,
                )
                bootstrap_operation_parameter_defs(conn)
                _drop_legacy_tables(conn)
                return

            # Existing database. Per PRD-19 (option B: MFDB is unreleased and
            # pre-PRD-19 data is disposable) there is no versioned migration
            # waterfall. Build any missing canonical tables (CREATE IF NOT
            # EXISTS is idempotent) and reconcile the live schema to the
            # dictionary; the post-processing below seeds vocabulary and drops
            # legacy/duplicate tables. An incompatible pre-PRD-19 database can
            # simply be deleted and recreated.
            # Create any missing tables first (CREATE IF NOT EXISTS is a no-op for
            # tables that already exist; it does NOT add columns to them).
            for sql in FRESH_DB_TABLES_SQL:
                cursor.execute(sql)
            set_schema_version(conn, SCHEMA_VERSION)
            # Then ensure every canonical column (incl. flr_*) exists on existing
            # tables, and reconcile mfdb_* extension tables, BEFORE creating indices
            # (an index on a not-yet-added column would fail on an older DB).
            _ensure_canonical_columns(conn)
            from chisurf.core.mfdb.schema_from_dictionary import reconcile_schema
            from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
            reconcile_schema(conn, MmcifDictionary.load_bundled())
            _ensure_mfdb_setup_columns(conn)
            _ensure_lifecycle_columns(conn)
            # Indices + triggers last; tolerate an odd missing column on a stale DB.
            for sql in FRESH_DB_INDICES_SQL + MFDB_EDGE_VOCABULARY_TRIGGER_SQL:
                try:
                    cursor.execute(sql)
                except sqlite3.OperationalError:
                    pass

        finally:
            cursor.close()
    bootstrap_vocabulary(conn)
    from chisurf.core.mfdb.operation_parameters import bootstrap_operation_parameter_defs
    bootstrap_operation_parameter_defs(conn)
    # Ensure auth columns exist on flr_sample_users (for DBs that skipped v22 migration)
    for col, col_type in [("is_admin", "INTEGER DEFAULT 0"), ("password_hash", "TEXT"), ("allow_passwordless_login", "INTEGER DEFAULT 0")]:
        with conn:
            _ensure_column(conn, "flr_sample_users", col, col_type)
    bootstrap_default_user(conn)
    try:
        bootstrap_auth_groups(conn)
    except sqlite3.OperationalError:
        pass
    # Phase 3: drop duplicate/legacy tables
    _drop_legacy_tables(conn)
    return report


def _parse_create_table_columns(create_sql: str) -> tuple[str | None, list[tuple[str, str]]]:
    """Extract ``(table_name, [(column_name, column_def), ...])`` from a CREATE.

    Constraint clauses (PRIMARY KEY / FOREIGN KEY / UNIQUE / CHECK / CONSTRAINT)
    are skipped — only real column definitions are returned.
    """
    m = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*)\)\s*$", create_sql, re.S)
    if not m:
        return None, []
    table, body = m.group(1), m.group(2)
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    constraint_kw = {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}
    cols: list[tuple[str, str]] = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        tokens = s.split()
        if tokens[0].upper() in constraint_kw:
            continue
        cols.append((tokens[0].strip('"'), s))
    return table, cols


def _ensure_canonical_columns(conn: sqlite3.Connection) -> None:
    """ALTER-add any canonical column missing from an existing live table.

    Option B (PRD-19) replaced the versioned migration waterfall with
    ``reconcile_schema``, which only covers ``mfdb_*`` extension tables. This adds
    missing columns to the remaining canonical/``flr_*`` tables too, so an existing
    database (notably the shipped curated source DB) is brought to the current
    schema rather than failing on a missing column.
    """
    cur = conn.cursor()
    try:
        live = {
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for sql in FRESH_DB_TABLES_SQL:
            table, cols = _parse_create_table_columns(sql)
            if not table or table not in live:
                continue
            existing = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, coldef in cols:
                if name in existing:
                    continue
                # SQLite forbids PRIMARY KEY / UNIQUE in ALTER ADD COLUMN.
                add_def = re.sub(r"\bPRIMARY KEY\b|\bUNIQUE\b", "", coldef, flags=re.I).strip()
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {add_def}")
                except sqlite3.OperationalError:
                    # Non-constant DEFAULT (e.g. CURRENT_TIMESTAMP) or other clause
                    # rejected by ALTER: fall back to a bare typed column.
                    tokens = coldef.split()
                    coltype = tokens[1] if len(tokens) > 1 and tokens[1].upper() not in {
                        "PRIMARY", "REFERENCES", "DEFAULT", "NOT", "UNIQUE", "CHECK"
                    } else "TEXT"
                    try:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")
                    except sqlite3.OperationalError:
                        pass
    finally:
        cur.close()


def _drop_legacy_tables(conn: sqlite3.Connection) -> None:
    """Drop duplicate (mfdb_sample, mfdb_experiment) and legacy (fdb_*) tables."""
    legacy = [
        "mfdb_sample", "mfdb_experiment",
        "fdb_raw_data", "fdb_processing_run", "fdb_processing_input",
        "fdb_processed_data", "fdb_provenance_edge", "fdb_setup_definition",
        "fdb_analysis_run", "fdb_analysis_parameter", "fdb_audit_log",
        "fdb_setup", "fdb_artifact", "fdb_operation",
        "fdb_operation_artifact", "fdb_edge", "fdb_parameter",
    ]
    with conn:
        for table in legacy:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            except sqlite3.OperationalError:
                pass  # table may not exist; that's fine


