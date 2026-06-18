import json as _json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

SCHEMA_VERSION = 28


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
        setup_definition_id TEXT REFERENCES fdb_setup_definition(setup_id) ON DELETE SET NULL,
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
        UNIQUE (donor_probe_id, acceptor_probe_id)
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
    """CREATE TABLE IF NOT EXISTS fdb_raw_data (
        raw_data_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        data_type TEXT NOT NULL,
        storage_mode TEXT NOT NULL,
        file_path TEXT,
        url TEXT,
        folder_path TEXT,
        mime_type TEXT,
        size_bytes INTEGER,
        checksum TEXT,
        checksum_algorithm TEXT DEFAULT 'sha256',
        header_metadata_json TEXT,
        detector_mapping_json TEXT,
        acquisition_software TEXT,
        acquisition_software_version TEXT,
        acquired_at TEXT,
        validation_status TEXT DEFAULT 'unvalidated',
        validation_message TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_processing_run (
        processing_id TEXT PRIMARY KEY,
        processing_type TEXT NOT NULL,
        experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        settings_json TEXT,
        settings_hash TEXT,
        selected_setup_name TEXT,
        detector_definitions_json TEXT,
        pie_window_definitions_json TEXT,
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
        file_count INTEGER,
        photon_count INTEGER,
        selected_photon_count INTEGER,
        burst_count INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_processing_input (
        processing_id TEXT NOT NULL REFERENCES fdb_processing_run(processing_id) ON DELETE CASCADE,
        raw_data_id TEXT NOT NULL REFERENCES fdb_raw_data(raw_data_id) ON DELETE CASCADE,
        ordinal INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        PRIMARY KEY (processing_id, raw_data_id)
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_processed_data (
        processed_data_id TEXT PRIMARY KEY,
        processing_id TEXT NOT NULL REFERENCES fdb_processing_run(processing_id) ON DELETE CASCADE,
        product_type TEXT NOT NULL,
        storage_mode TEXT NOT NULL,
        file_path TEXT,
        url TEXT,
        folder_path TEXT,
        mime_type TEXT,
        size_bytes INTEGER,
        checksum TEXT,
        checksum_algorithm TEXT DEFAULT 'sha256',
        row_count INTEGER,
        product_summary_json TEXT,
        metadata_json TEXT,
        data_json TEXT,
        data_blob BLOB,
        validation_status TEXT DEFAULT 'unvalidated',
        validation_message TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_provenance_edge (
        edge_id INTEGER PRIMARY KEY,
        source_node_type TEXT NOT NULL,
        source_node_id TEXT NOT NULL,
        target_node_type TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        processing_id TEXT,
        settings_hash TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        software_version TEXT,
        checksum_snapshot_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_setup_definition (
        setup_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version INTEGER DEFAULT 1,
        instrument_id TEXT REFERENCES flr_instrument(instrument_id),
        description TEXT,
        configuration_json TEXT,
        detectors_json TEXT,
        timing_calibration_json TEXT,
        irf_definition_json TEXT,
        burst_defaults_json TEXT,
        fcs_calibration_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_analysis_run (
        analysis_id TEXT PRIMARY KEY REFERENCES fdb_processing_run(processing_id) ON DELETE CASCADE,
        model_name TEXT,
        model_type TEXT,
        model_version TEXT,
        fit_structure_json TEXT,
        parameter_links_json TEXT,
        covariance_matrix_json TEXT,
        goodness_of_fit_json TEXT,
        notes TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_analysis_parameter (
        parameter_id INTEGER PRIMARY KEY AUTOINCREMENT,
        parameter_uuid TEXT UNIQUE NOT NULL,
        analysis_id TEXT REFERENCES fdb_analysis_run(analysis_id) ON DELETE CASCADE,
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
        expression TEXT,
        prior_json TEXT,
        mapping_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_audit_log (
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
    # Legacy fdb_setup (preserve as source for v17 backfill; canonical target is mfdb_setup)
    """CREATE TABLE IF NOT EXISTS fdb_setup (
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
        burst_defaults_json TEXT,
        fcs_calibration_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    # Canonical FDB Tables (v17 target architecture — preserved for migration backfill compat)
    """CREATE TABLE IF NOT EXISTS fdb_artifact (
        artifact_id TEXT PRIMARY KEY,
        artifact_type TEXT NOT NULL,
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
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_operation (
        operation_id TEXT PRIMARY KEY,
        operation_type TEXT NOT NULL,
        experiment_id TEXT REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        setup_id TEXT REFERENCES fdb_setup(setup_id) ON DELETE SET NULL,
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
    """CREATE TABLE IF NOT EXISTS fdb_operation_artifact (
        operation_id TEXT NOT NULL REFERENCES fdb_operation(operation_id) ON DELETE CASCADE,
        artifact_id TEXT NOT NULL REFERENCES fdb_artifact(artifact_id) ON DELETE CASCADE,
        direction TEXT NOT NULL,
        role TEXT,
        ordinal INTEGER DEFAULT 0,
        checksum_snapshot TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT,
        PRIMARY KEY (operation_id, artifact_id, direction)
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_edge (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_node_type TEXT NOT NULL,
        source_node_id TEXT NOT NULL,
        target_node_type TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        operation_id TEXT REFERENCES fdb_operation(operation_id) ON DELETE SET NULL,
        settings_hash TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        software_version TEXT,
        checksum_snapshot_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_parameter (
        parameter_id INTEGER PRIMARY KEY AUTOINCREMENT,
        parameter_uuid TEXT UNIQUE NOT NULL,
        operation_id TEXT REFERENCES fdb_operation(operation_id) ON DELETE CASCADE,
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
        expression TEXT,
        prior_json TEXT,
        mapping_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
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
        burst_defaults_json TEXT,
        fcs_calibration_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
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
    """CREATE TABLE IF NOT EXISTS mfdb_sample (
        sample_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        sample_type TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS mfdb_experiment (
        experiment_id TEXT PRIMARY KEY,
        sample_id TEXT REFERENCES mfdb_sample(sample_id) ON DELETE SET NULL,
        display_name TEXT NOT NULL,
        project_id TEXT,
        status TEXT DEFAULT 'pending',
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
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
        expression TEXT,
        prior_json TEXT,
        mapping_json TEXT,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT
    )""",
]

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
    if "CREATE TABLE IF NOT EXISTS fdb_" in sql:
        continue
    for mfdb_name, replacement in _CANONICAL_TABLE_MAP.items():
        prefix = f"CREATE TABLE IF NOT EXISTS {mfdb_name} "
        if prefix in sql:
            sql = replacement
            break
    sql = sql.replace("REFERENCES fdb_setup_definition(setup_id)", "REFERENCES mfdb_setup(setup_id)")
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
    "CREATE INDEX IF NOT EXISTS idx_fdb_raw_data_experiment ON fdb_raw_data (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_processing_run_experiment ON fdb_processing_run (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_processing_run_type ON fdb_processing_run (processing_type)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_processed_data_processing ON fdb_processed_data (processing_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_processed_data_type ON fdb_processed_data (product_type)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_provenance_source ON fdb_provenance_edge (source_node_type, source_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_provenance_target ON fdb_provenance_edge (target_node_type, target_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_provenance_processing ON fdb_provenance_edge (processing_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_setup_definition_instrument ON fdb_setup_definition (instrument_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_setup ON flr_experiment (setup_definition_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_analysis_parameter_analysis ON fdb_analysis_parameter (analysis_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_analysis_parameter_uuid ON fdb_analysis_parameter (parameter_uuid)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_audit_log_target ON fdb_audit_log (target_type, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_fdb_audit_log_timestamp ON fdb_audit_log (timestamp)",
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
    "CREATE INDEX IF NOT EXISTS idx_mfdb_audit_log_target ON mfdb_audit_log (target_type, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_audit_log_timestamp ON mfdb_audit_log (timestamp)",
    # New MFDB table indices
    "CREATE INDEX IF NOT EXISTS idx_mfdb_sample_type ON mfdb_sample (sample_type)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_experiment_sample ON mfdb_experiment (sample_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_experiment_project ON mfdb_experiment (project_id)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_experiment_status ON mfdb_experiment (status)",
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
    "CREATE INDEX IF NOT EXISTS idx_mfdb_sample_deleted_at ON mfdb_sample (deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_mfdb_experiment_deleted_at ON mfdb_experiment (deleted_at)",
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
        ("fdb_raw_data", True, True),
        ("fdb_processing_run", True, True),
        ("fdb_processed_data", True, True),
        ("fdb_setup_definition", True, True),
        ("fdb_analysis_run", True, True),
        ("fdb_analysis_parameter", True, True),
        ("fdb_setup", True, True),
        ("fdb_artifact", True, True),
        ("fdb_operation", True, True),
        ("fdb_parameter", True, True),
        ("mfdb_artifact", True, True),
        ("mfdb_operation", True, True),
        ("mfdb_parameter", True, True),
        ("mfdb_setup", True, True),
        ("mfdb_sample", True, True),
        ("mfdb_experiment", True, True),
        ("mfdb_branch", True, True),
        ("fdb_audit_log", True, False),
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
        ("fdb_processing_input", False, False),
        ("fdb_provenance_edge", False, False),
        ("fdb_operation_artifact", False, False),
        ("fdb_edge", False, False),
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
    """Bootstrap built-in and legacy extensible vocabulary values in mfdb_vocabulary."""
    # Extensible vocabulary values to seed
    vocab = {
        "artifact_kind": [
            "raw_measurement", "processed_data", "analysis_result",
            "fit_result", "parameter_table", "selection_mask",
            "project_snapshot", "archive_manifest", "archive_file",
            "visualization", "external_reference", "chinet_session",
            "chinet_node",
            # Legacy
            "raw_data", "bur", "ptu", "spc", "bh", "fcs",
            "tcspc", "decay", "irf", "pda", "model_curve", "residual",
            "plot_export", "table_export", "project_archive", "external_file",
            "photon_hdf5", "burst_table", "spectra", "hdf5", "zip",
            "json_summary", "mti_summary", "fcs_correlation", "irf_curve",
            "tcspc_decay", "anisotropy_curve", "pda_histogram", "fit_results",
            "derived_product", "gmm_summary", "clustering_labels"
        ],
        "data_format": [
            "ptu", "spc", "bh", "tttr", "photon_hdf5", "bur",
            "hdf5", "zip", "json", "csv", "tsv", "png", "svg",
            "sqlite", "directory", "unknown"
        ],
        "operation_type": [
            "measurement_import", "validation", "burst_selection",
            "filtering", "fcs_correlation", "microtime_histogram",
            "tcspc_fitting", "model_fitting", "ndxplorer_selection",
            "ndxplorer_clustering", "project_snapshot", "project_restore",
            "archive_export",
            "tcspc_histogram_computation", "pda_histogram_computation",
            "pch_histogram_computation", "fcs_correlation_load",
            "tcspc_curve_load",
            # Legacy/Custom
            "import", "burst_filtering", "gmm_fitting", "analysis",
            "fitting", "project_archive", "local_fit", "global_fit",
            "project", "analysis_run", "decay_fit"
        ],
        "parameter_type": [
            "free", "fixed", "linked", "shared", "local", "global", "calibrated"
        ],
        "relationship_type": [
            "included_in", "contains", "derived_from", "supersedes",
            "uses_external_reference", "parameter_depends_on", "parameter_of", "linked_to",
            "project_contains", "grouped_in", "measured_sample"
        ],
        # Sample-related vocabularies (PRD-02 Task 6)
        "entity_type": [
            "protein", "dna", "rna", "polymer", "non-polymer",
            "water", "macromolecule", "oligosaccharide", "ligand", "solvent"
        ],
        "fluorophore_type": [
            "donor", "acceptor", "unspecified"
        ],
        "solvent_phase": [
            "liquid", "solid", "gas", "vitrified"
        ],
        "sample_type": [
            "protein", "dna", "rna", "physical_sample"
        ],
        "probe_origin": [
            "extrinsic", "intrinsic"
        ],
        "probe_link_type": [
            "covalent", "non-covalent", "genetic"
        ],
        "reactive_probe_flag": [
            "yes", "no"
        ],
        "ambiguous_stoichiometry": [
            "yes", "no"
        ],
    }
    with conn:
        for field_name, values in vocab.items():
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
                return

            # Read version, start legacy migration waterfall
            version = get_schema_version(conn)

            if "flr_experiment" in existing:
                t_cols = {
                    r[1] for r in cursor.execute("PRAGMA table_info(flr_experiment)").fetchall()
                }
                if "id" in t_cols and "experiment_id" not in t_cols:
                    cursor.execute("ALTER TABLE flr_experiment RENAME COLUMN id TO experiment_id")

            # Phase 1: Legacy -> v1
            if version < 1:
                if "item_types" in existing and "probe_types" not in existing:
                    cursor.execute("ALTER TABLE item_types RENAME TO probe_types")
                    tp_cols = {
                        r[1] for r in cursor.execute("PRAGMA table_info(probe_types)").fetchall()
                    }
                    if "id" in tp_cols and "type_id" not in tp_cols:
                        cursor.execute("ALTER TABLE probe_types RENAME COLUMN id TO type_id")
                    if "name" in tp_cols and "type_name" not in tp_cols:
                        cursor.execute("ALTER TABLE probe_types RENAME COLUMN name TO type_name")

                if "items" in existing and "probes" not in existing:
                    cursor.execute("ALTER TABLE items RENAME TO probes")

                cols = {r[1] for r in cursor.execute("PRAGMA table_info(probes)").fetchall()}
                if "id" in cols and "probe_id" not in cols:
                    cursor.execute("ALTER TABLE probes RENAME COLUMN id TO probe_id")
                if "name" in cols and "chromophore_name" not in cols:
                    cursor.execute("ALTER TABLE probes RENAME COLUMN name TO chromophore_name")

                to_add = {
                    "category": "TEXT DEFAULT 'other'",
                    "quality_flag": "INTEGER DEFAULT 1",
                    "is_curated": "INTEGER DEFAULT 0",
                    "probe_origin": "TEXT DEFAULT 'extrinsic'",
                    "probe_link_type": "TEXT DEFAULT 'covalent'",
                    "fluorophore_type": "TEXT DEFAULT 'unspecified'",
                    "reactive_probe_flag": "TEXT DEFAULT 'no'",
                    "reactive_probe_name": "TEXT",
                    "chromophore_chem_descriptor_id": "INTEGER",
                    "reactive_probe_chem_descriptor_id": "INTEGER",
                    "chromophore_center_atom": "TEXT",
                    "description": "TEXT DEFAULT ''",
                }
                for col, definition in to_add.items():
                    if col not in cols:
                        cursor.execute(f"ALTER TABLE probes ADD COLUMN {col} {definition}")

                for table in ["spectra", "optical_properties"]:
                    if table in existing:
                        t_cols = {
                            r[1] for r in cursor.execute(f"PRAGMA table_info({table})").fetchall()
                        }
                        if "item_id" in t_cols and "probe_id" not in t_cols:
                            cursor.execute(f"ALTER TABLE {table} ADD COLUMN probe_id INTEGER")
                            cursor.execute(f"UPDATE {table} SET probe_id = item_id")
                        if table == "optical_properties" and "unit" not in t_cols:
                            cursor.execute("ALTER TABLE optical_properties ADD COLUMN unit TEXT")

                cursor.execute("CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER)")
                set_schema_version(conn, 1)
                version = 1

            if version < 2:
                for sql in CREATE_TABLES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, 2)
                version = 2

            if version < 3:
                for sql in CREATE_INDICES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, 3)
                version = 3

            if version < 4:
                set_schema_version(conn, 4)
                version = 4

            if version < 5:
                tp_cols = {
                    r[1] for r in cursor.execute("PRAGMA table_info(probe_types)").fetchall()
                }
                if "id" in tp_cols and "type_id" not in tp_cols:
                    cursor.execute("ALTER TABLE probe_types RENAME COLUMN id TO type_id")
                if "name" in tp_cols and "type_name" not in tp_cols:
                    cursor.execute("ALTER TABLE probe_types RENAME COLUMN name TO type_name")

                p_cols = {r[1] for r in cursor.execute("PRAGMA table_info(probes)").fetchall()}
                if "id" in p_cols and "probe_id" not in p_cols:
                    cursor.execute("ALTER TABLE probes RENAME COLUMN id TO probe_id")

                if "name" in p_cols and "chromophore_name" in p_cols:
                    cursor.execute(
                        "UPDATE probes SET chromophore_name = name WHERE chromophore_name IS NULL OR chromophore_name = ''"
                    )
                elif "name" in p_cols and "chromophore_name" not in p_cols:
                    cursor.execute("ALTER TABLE probes RENAME COLUMN name TO chromophore_name")

                set_schema_version(conn, 5)
                version = 5

            if version < 6:
                for table in ["images", "spectra", "optical_properties"]:
                    if table in existing:
                        t_cols = {
                            r[1] for r in cursor.execute(f"PRAGMA table_info({table})").fetchall()
                        }
                        if "item_id" in t_cols and "probe_id" not in t_cols:
                            cursor.execute(f"ALTER TABLE {table} ADD COLUMN probe_id INTEGER")
                            cursor.execute(f"UPDATE {table} SET probe_id = item_id")
                set_schema_version(conn, 6)
                version = 6

            if version < 7:
                for sql in CREATE_TABLES_SQL:
                    cursor.execute(sql)

                _ensure_column(cursor, "spectra", "wavelength_unit", "TEXT DEFAULT 'nm'")
                _ensure_column(cursor, "spectra", "intensity_unit", "TEXT DEFAULT 'normalized'")
                _ensure_column(cursor, "spectra", "details", "TEXT")
                _ensure_column(cursor, "optical_properties", "details", "TEXT")
                _ensure_column(cursor, "flr_sample", "num_of_probes", "INTEGER")
                _ensure_column(cursor, "flr_sample", "solvent_phase", "TEXT")
                _ensure_column(cursor, "flr_sample", "sample_condition_id", "TEXT")
                _ensure_column(cursor, "flr_sample", "entity_assembly_id", "TEXT")
                _ensure_column(cursor, "flr_poly_probe_position", "atom_id", "TEXT")
                _ensure_column(cursor, "flr_poly_probe_position", "mutation_flag", "TEXT DEFAULT 'no'")
                _ensure_column(cursor, "flr_poly_probe_position", "modification_flag", "TEXT DEFAULT 'no'")
                _ensure_column(cursor, "flr_poly_probe_position", "auth_name", "TEXT")
                _ensure_column(cursor, "flr_fret_forster_radius", "reduced_forster_radius", "REAL")
                _ensure_column(cursor, "flr_fret_analysis", "experiment_id", "TEXT")
                _ensure_column(cursor, "flr_fret_analysis", "type", "TEXT")
                _ensure_column(cursor, "flr_fret_analysis", "sample_probe_id_1", "INTEGER")
                _ensure_column(cursor, "flr_fret_analysis", "sample_probe_id_2", "INTEGER")
                _ensure_column(cursor, "flr_fret_analysis", "forster_radius_id", "INTEGER")
                _ensure_column(cursor, "flr_fret_analysis", "dataset_list_id", "INTEGER")
                _ensure_column(cursor, "flr_fret_analysis", "external_file_id", "INTEGER")
                _ensure_column(cursor, "flr_fret_analysis", "software_id", "INTEGER")
                _ensure_column(cursor, "flr_fret_distance_restraint", "state_id", "INTEGER")
                _ensure_column(cursor, "flr_fret_distance_restraint", "population_fraction", "REAL")
                _ensure_column(cursor, "flr_fret_distance_restraint", "peak_assignment_id", "INTEGER")

                cursor.execute("UPDATE spectra SET wavelength_unit = 'nm' WHERE wavelength_unit IS NULL")
                cursor.execute("UPDATE spectra SET intensity_unit = 'normalized' WHERE intensity_unit IS NULL")
                cursor.execute("UPDATE flr_poly_probe_position SET mutation_flag = 'no' WHERE mutation_flag IS NULL")
                cursor.execute("UPDATE flr_poly_probe_position SET modification_flag = 'no' WHERE modification_flag IS NULL")
                cursor.execute("UPDATE flr_fret_forster_radius SET reduced_forster_radius = forster_radius WHERE reduced_forster_radius IS NULL AND forster_radius IS NOT NULL")

                for sql in CREATE_INDICES_SQL:
                    cursor.execute(sql)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_metadata_analysis ON analysis_metadata (analysis_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_photon_stream_analysis ON flr_photon_stream (analysis_id)")
                _ensure_column(cursor, "ihm_external_files", "md5", "TEXT")
                _ensure_column(cursor, "ihm_external_files", "uuid", "TEXT")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_external_files_path ON ihm_external_files (file_path)")

                set_schema_version(conn, 7)
                version = 7

            if version < 8:
                _ensure_column(cursor, "flr_sample", "sample_uuid", "TEXT")
                cursor.execute("UPDATE flr_sample SET sample_uuid = hex(randomblob(16)) WHERE sample_uuid IS NULL")
                set_schema_version(conn, 8)
                version = 8

            if version < 9:
                cursor.execute("CREATE TABLE IF NOT EXISTS flr_sample_probe (sample_probe_id INTEGER PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE, probe_id INTEGER NOT NULL REFERENCES probes(probe_id), poly_probe_position_id INTEGER REFERENCES flr_poly_probe_position(id), fluorophore_type TEXT DEFAULT 'unspecified', description TEXT, UNIQUE (sample_id, probe_id, poly_probe_position_id))")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sample_probe_sample ON flr_sample_probe (sample_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sample_probe_probe ON flr_sample_probe (probe_id)")
                set_schema_version(conn, 9)
                version = 9

            if version < 10:
                cursor.execute("CREATE TABLE IF NOT EXISTS analysis_data (id INTEGER PRIMARY KEY, analysis_id TEXT NOT NULL REFERENCES flr_fret_analysis(analysis_id), data_type TEXT NOT NULL, data_name TEXT, x_values BLOB NOT NULL, y_values BLOB NOT NULL, x_unit TEXT, y_unit TEXT, details TEXT)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_data_analysis ON analysis_data (analysis_id)")
                set_schema_version(conn, 10)
                version = 10

            if version < 11:
                if "flr_sample" in existing:
                    for column, definition in {
                        "project_id": "TEXT",
                        "measured_by_user_id": "TEXT",
                        "measured_by_device_id": "TEXT",
                        "measured_at": "TEXT",
                    }.items():
                        _ensure_column(cursor, "flr_sample", column, definition)
                cursor.execute("CREATE TABLE IF NOT EXISTS flr_sample_users (user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, email TEXT, affiliation TEXT, details TEXT)")
                cursor.execute("CREATE TABLE IF NOT EXISTS flr_sample_devices (device_id TEXT PRIMARY KEY, name TEXT NOT NULL, device_type TEXT, model TEXT, serial_number TEXT, location TEXT, owner TEXT, details TEXT)")
                cursor.execute("CREATE TABLE IF NOT EXISTS flr_sample_key_value (sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT, details TEXT, UNIQUE (sample_id, key))")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sample_key_value_sample ON flr_sample_key_value (sample_id)")
                set_schema_version(conn, 11)
                version = 11

            if version < 12:
                if "flr_experiment" in existing:
                    t_cols = {
                        r[1] for r in cursor.execute("PRAGMA table_info(flr_experiment)").fetchall()
                    }
                    if "id" in t_cols and "experiment_id" not in t_cols:
                        cursor.execute("ALTER TABLE flr_experiment RENAME COLUMN id TO experiment_id")
                    for column, definition in {
                        "type_id": "INTEGER",
                        "sample_id": "TEXT",
                        "project_id": "TEXT",
                        "measured_by_user_id": "TEXT",
                        "measured_by_device_id": "TEXT",
                        "started_at": "TEXT",
                        "ended_at": "TEXT",
                        "status": "TEXT",
                    }.items():
                        _ensure_column(cursor, "flr_experiment", column, definition)
                cursor.execute("CREATE TABLE IF NOT EXISTS flr_experiment_type (type_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, category TEXT, description TEXT, details TEXT)")
                cursor.execute("CREATE TABLE IF NOT EXISTS flr_experiment_key_value (experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT, details TEXT, UNIQUE (experiment_id, key))")
                cursor.execute("CREATE TABLE IF NOT EXISTS flr_experiment_data (data_id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE, data_type TEXT NOT NULL, storage_mode TEXT NOT NULL, file_path TEXT, url TEXT, folder_path TEXT, mime_type TEXT, size_bytes INTEGER, checksum TEXT, data_json TEXT, data_blob BLOB, reading_options_json TEXT, details TEXT)")
                for sql in CREATE_INDICES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, 12)
                version = 12

            if version < 13:
                for sql in CREATE_TABLES_SQL:
                    if "fdb_" in sql:
                        cursor.execute(sql)
                for sql in CREATE_INDICES_SQL:
                    if "fdb_" in sql:
                        cursor.execute(sql)
                set_schema_version(conn, 13)
                version = 13

            if version < 14:
                for sql in CREATE_TABLES_SQL:
                    if "fdb_setup_definition" in sql:
                        cursor.execute(sql)
                _ensure_column(cursor, "flr_experiment", "setup_definition_id", "TEXT REFERENCES fdb_setup_definition(setup_id) ON DELETE SET NULL")
                for sql in CREATE_INDICES_SQL:
                    if "idx_fdb_setup_definition" in sql or "idx_experiment_setup" in sql:
                        cursor.execute(sql)
                set_schema_version(conn, 14)
                version = 14

            if version < 15:
                for sql in CREATE_TABLES_SQL:
                    if "fdb_analysis_run" in sql or "fdb_analysis_parameter" in sql:
                        cursor.execute(sql)
                for sql in CREATE_INDICES_SQL:
                    if "idx_fdb_analysis_run" in sql or "idx_fdb_analysis_parameter" in sql:
                        cursor.execute(sql)
                set_schema_version(conn, 15)
                version = 15

            if version < 16:
                for sql in CREATE_TABLES_SQL:
                    if "fdb_audit_log" in sql:
                        cursor.execute(sql)
                for sql in CREATE_INDICES_SQL:
                    if "idx_fdb_audit_log" in sql:
                        cursor.execute(sql)
                set_schema_version(conn, 16)
                version = 16

            # Phase 17: Canonical Schema Migration & Python-based Backfill
            if version < 17:
                logger.info("Migrating database schema to version 17 (canonical target architecture)...")
                for sql in CREATE_TABLES_SQL:
                    if "fdb_artifact" in sql or "fdb_operation" in sql or "fdb_operation_artifact" in sql or "fdb_edge" in sql or "fdb_parameter" in sql or "fdb_setup" in sql:
                        cursor.execute(sql)

                for sql in CREATE_INDICES_SQL:
                    if any(t in sql for t in ["fdb_artifact", "fdb_operation", "fdb_edge", "fdb_parameter", "fdb_setup"]):
                        cursor.execute(sql)

                conn.row_factory = sqlite3.Row

                # 3.1 Backfill fdb_raw_data -> fdb_artifact
                raw_rows = conn.execute("SELECT * FROM fdb_raw_data").fetchall()
                for row in raw_rows:
                    meta = {}
                    if row["header_metadata_json"]:
                        try:
                            meta.update(_json.loads(row["header_metadata_json"]))
                        except Exception:
                            pass
                    if row["detector_mapping_json"]:
                        try:
                            meta["detector_mapping"] = _json.loads(row["detector_mapping_json"])
                        except Exception:
                            pass
                    meta["acquisition_software"] = row["acquisition_software"]
                    meta["acquisition_software_version"] = row["acquisition_software_version"]
                    meta["acquired_at"] = row["acquired_at"]

                    conn.execute(
                        """INSERT OR IGNORE INTO fdb_artifact (
                            artifact_id, artifact_type, experiment_id, storage_mode, file_path, url, folder_path,
                            mime_type, size_bytes, checksum, checksum_algorithm, metadata_json,
                            validation_status, validation_message, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row["raw_data_id"],
                            "raw_data",
                            row["experiment_id"],
                            row["storage_mode"],
                            row["file_path"],
                            row["url"],
                            row["folder_path"],
                            row["mime_type"],
                            row["size_bytes"],
                            row["checksum"],
                            row["checksum_algorithm"] or "sha256",
                            _json.dumps(meta),
                            row["validation_status"],
                            row["validation_message"],
                            row["created_at"],
                            row["updated_at"],
                        )
                    )

                # 3.2 Backfill fdb_processed_data -> fdb_artifact
                proc_data_rows = conn.execute("SELECT * FROM fdb_processed_data").fetchall()
                for row in proc_data_rows:
                    meta = {}
                    if row["metadata_json"]:
                        try:
                            meta.update(_json.loads(row["metadata_json"]))
                        except Exception:
                            pass
                    if row["product_summary_json"]:
                        try:
                            meta["product_summary"] = _json.loads(row["product_summary_json"])
                        except Exception:
                            pass

                    conn.execute(
                        """INSERT OR IGNORE INTO fdb_artifact (
                            artifact_id, artifact_type, storage_mode, file_path, url, folder_path,
                            mime_type, size_bytes, checksum, checksum_algorithm, row_count,
                            validation_status, validation_message, metadata_json, data_json, data_blob,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row["processed_data_id"],
                            row["product_type"],
                            row["storage_mode"],
                            row["file_path"],
                            row["url"],
                            row["folder_path"],
                            row["mime_type"],
                            row["size_bytes"],
                            row["checksum"],
                            row["checksum_algorithm"] or "sha256",
                            row["row_count"],
                            row["validation_status"],
                            row["validation_message"],
                            _json.dumps(meta),
                            row["data_json"],
                            row["data_blob"],
                            row["created_at"],
                            row["updated_at"],
                        )
                    )

                # 3.3 Backfill fdb_setup_definition -> fdb_setup
                setup_rows = conn.execute("SELECT * FROM fdb_setup_definition").fetchall()
                for row in setup_rows:
                    conn.execute(
                        """INSERT OR IGNORE INTO fdb_setup (
                            setup_id, name, version, instrument_id, description, configuration_json,
                            detectors_json, timing_calibration_json, irf_definition_json, burst_defaults_json,
                            fcs_calibration_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row["setup_id"],
                            row["name"],
                            row["version"] or 1,
                            row["instrument_id"],
                            row["description"],
                            row["configuration_json"],
                            row["detectors_json"],
                            row["timing_calibration_json"],
                            row["irf_definition_json"],
                            row["burst_defaults_json"],
                            row["fcs_calibration_json"],
                            row["created_at"],
                            row["updated_at"],
                        )
                    )

                # 3.4 Backfill fdb_processing_run -> fdb_operation
                run_rows = conn.execute("SELECT * FROM fdb_processing_run").fetchall()
                for row in run_rows:
                    meta = {}
                    meta["selected_setup_name"] = row["selected_setup_name"]
                    if row["detector_definitions_json"]:
                        try:
                            meta["detector_definitions"] = _json.loads(row["detector_definitions_json"])
                        except Exception:
                            pass
                    if row["pie_window_definitions_json"]:
                        try:
                            meta["pie_window_definitions"] = _json.loads(row["pie_window_definitions_json"])
                        except Exception:
                            pass
                    meta["file_count"] = row["file_count"]
                    meta["photon_count"] = row["photon_count"]
                    meta["selected_photon_count"] = row["selected_photon_count"]
                    meta["burst_count"] = row["burst_count"]

                    conn.execute(
                        """INSERT OR IGNORE INTO fdb_operation (
                            operation_id, operation_type, experiment_id, settings_json, settings_hash,
                            operator_user_id, software_package, software_module, software_version,
                            runtime_environment_json, started_at, ended_at, status, error_message,
                            traceback_summary, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row["processing_id"],
                            row["processing_type"],
                            row["experiment_id"],
                            row["settings_json"],
                            row["settings_hash"],
                            row["operator_user_id"],
                            row["software_package"],
                            row["software_module"],
                            row["software_version"],
                            row["runtime_environment_json"],
                            row["started_at"],
                            row["ended_at"],
                            row["status"],
                            row["error_message"],
                            row["traceback_summary"],
                            _json.dumps(meta),
                            row["created_at"],
                            row["updated_at"],
                        )
                    )

                # 3.5 Backfill fdb_analysis_run -> fdb_operation (supplement existing or insert new)
                analysis_rows = conn.execute("SELECT * FROM fdb_analysis_run").fetchall()
                for row in analysis_rows:
                    meta = {}
                    meta["model_name"] = row["model_name"]
                    meta["model_type"] = row["model_type"]
                    meta["model_version"] = row["model_version"]
                    if row["covariance_matrix_json"]:
                        try:
                            meta["covariance_matrix"] = _json.loads(row["covariance_matrix_json"])
                        except Exception:
                            pass
                    if row["goodness_of_fit_json"]:
                        try:
                            meta["goodness_of_fit"] = _json.loads(row["goodness_of_fit_json"])
                        except Exception:
                            pass
                    meta["notes"] = row["notes"]
                    if row["metadata_json"]:
                        try:
                            meta.update(_json.loads(row["metadata_json"]))
                        except Exception:
                            pass

                    settings = {}
                    if row["fit_structure_json"]:
                        try:
                            settings["fit_structure"] = _json.loads(row["fit_structure_json"])
                        except Exception:
                            pass
                    if row["parameter_links_json"]:
                        try:
                            settings["parameter_links"] = _json.loads(row["parameter_links_json"])
                        except Exception:
                            pass

                    op_exists = conn.execute("SELECT 1 FROM fdb_operation WHERE operation_id = ?", (row["analysis_id"],)).fetchone()
                    if op_exists:
                        conn.execute(
                            """UPDATE fdb_operation SET
                                operation_type = 'analysis',
                                settings_json = ?,
                                metadata_json = ?
                                WHERE operation_id = ?""",
                            (
                                _json.dumps(settings) if settings else None,
                                _json.dumps(meta),
                                row["analysis_id"]
                            )
                        )
                    else:
                        conn.execute(
                            """INSERT OR IGNORE INTO fdb_operation (
                                operation_id, operation_type, settings_json, metadata_json, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                row["analysis_id"],
                                "analysis",
                                _json.dumps(settings) if settings else None,
                                _json.dumps(meta),
                                row["created_at"],
                                row["updated_at"],
                            )
                        )

                # 3.6 Backfill fdb_processing_input -> fdb_operation_artifact
                input_rows = conn.execute("SELECT * FROM fdb_processing_input").fetchall()
                for row in input_rows:
                    conn.execute(
                        """INSERT OR IGNORE INTO fdb_operation_artifact (
                            operation_id, artifact_id, direction, role, ordinal
                        ) VALUES (?, ?, ?, ?, ?)""",
                        (
                            row["processing_id"],
                            row["raw_data_id"],
                            "input",
                            "raw_data",
                            row["ordinal"] or 0,
                        )
                    )

                # 3.7 Backfill fdb_provenance_edge -> fdb_operation_artifact / fdb_edge
                edge_rows = conn.execute("SELECT * FROM fdb_provenance_edge").fetchall()
                for row in edge_rows:
                    conn.execute(
                        """INSERT OR IGNORE INTO fdb_edge (
                            edge_id, source_node_type, source_node_id, target_node_type, target_node_id,
                            relationship_type, operation_id, settings_hash, timestamp, software_version,
                            checksum_snapshot_json, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row["edge_id"],
                            row["source_node_type"],
                            row["source_node_id"],
                            row["target_node_type"],
                            row["target_node_id"],
                            row["relationship_type"],
                            row["processing_id"],
                            row["settings_hash"],
                            row["timestamp"],
                            row["software_version"],
                            row["checksum_snapshot_json"],
                            row["metadata_json"],
                        )
                    )

                    if row["relationship_type"] == "produced" and row["source_node_type"] == "processing_run" and row["target_node_type"] == "processed_data":
                        conn.execute(
                            """INSERT OR IGNORE INTO fdb_operation_artifact (
                                operation_id, artifact_id, direction, role
                            ) VALUES (?, ?, ?, ?)""",
                            (
                                row["source_node_id"],
                                row["target_node_id"],
                                "output",
                                "processed_data",
                            )
                        )
                    elif row["relationship_type"] == "input_to" and row["source_node_type"] == "processed_data" and row["target_node_type"] == "processing_run":
                        conn.execute(
                            """INSERT OR IGNORE INTO fdb_operation_artifact (
                                operation_id, artifact_id, direction, role
                            ) VALUES (?, ?, ?, ?)""",
                            (
                                row["target_node_id"],
                                row["source_node_id"],
                                "input",
                                "processed_data",
                            )
                        )

                # 3.8 Backfill fdb_analysis_parameter -> fdb_parameter
                param_rows = conn.execute("SELECT * FROM fdb_analysis_parameter").fetchall()
                for row in param_rows:
                    conn.execute(
                        """INSERT OR IGNORE INTO fdb_parameter (
                            parameter_id, parameter_uuid, operation_id, name, value, standard_error,
                            confidence_interval_low, confidence_interval_high, initial_value,
                            lower_bound, upper_bound, bounds_on, units, parameter_type, expression,
                            prior_json, mapping_json, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row["parameter_id"],
                            row["parameter_uuid"],
                            row["analysis_id"],
                            row["name"],
                            row["value"],
                            row["standard_error"],
                            row["confidence_interval_low"],
                            row["confidence_interval_high"],
                            row["initial_value"],
                            row["lower_bound"],
                            row["upper_bound"],
                            row["bounds_on"] or 0,
                            row["units"],
                            row["parameter_type"] or "free",
                            row["expression"],
                            row["prior_json"],
                            row["mapping_json"],
                            row["metadata_json"],
                            row["created_at"],
                            row["updated_at"],
                        )
                    )

                set_schema_version(conn, 17)
                version = 17
                logger.info("Successfully migrated schema and backfilled data to version 17.")
                report = MigrationReport(from_version=16, to_version=17)
                report.tables_added = [
                    t for t in ("fdb_artifact", "fdb_operation", "fdb_operation_artifact",
                                "fdb_edge", "fdb_parameter", "fdb_setup")
                ]
                source_counts = {
                    "fdb_raw_data": conn.execute("SELECT COUNT(*) FROM fdb_raw_data").fetchone()[0],
                    "fdb_processed_data": conn.execute("SELECT COUNT(*) FROM fdb_processed_data").fetchone()[0],
                    "fdb_setup_definition": conn.execute("SELECT COUNT(*) FROM fdb_setup_definition").fetchone()[0],
                    "fdb_processing_run": conn.execute("SELECT COUNT(*) FROM fdb_processing_run").fetchone()[0],
                    "fdb_analysis_run": conn.execute("SELECT COUNT(*) FROM fdb_analysis_run").fetchone()[0],
                    "fdb_processing_input": conn.execute("SELECT COUNT(*) FROM fdb_processing_input").fetchone()[0],
                    "fdb_provenance_edge": conn.execute("SELECT COUNT(*) FROM fdb_provenance_edge").fetchone()[0],
                    "fdb_analysis_parameter": conn.execute("SELECT COUNT(*) FROM fdb_analysis_parameter").fetchone()[0],
                }
                target_counts = {
                    "fdb_artifact": conn.execute("SELECT COUNT(*) FROM fdb_artifact").fetchone()[0],
                    "fdb_operation": conn.execute("SELECT COUNT(*) FROM fdb_operation").fetchone()[0],
                    "fdb_operation_artifact": conn.execute("SELECT COUNT(*) FROM fdb_operation_artifact").fetchone()[0],
                    "fdb_edge": conn.execute("SELECT COUNT(*) FROM fdb_edge").fetchone()[0],
                    "fdb_setup": conn.execute("SELECT COUNT(*) FROM fdb_setup").fetchone()[0],
                    "fdb_parameter": conn.execute("SELECT COUNT(*) FROM fdb_parameter").fetchone()[0],
                }
                report.backfill = {
                    "fdb_artifact (from raw_data)": {
                        "source": source_counts["fdb_raw_data"],
                        "inserted": target_counts["fdb_artifact"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "fdb_setup (from setup_definition)": {
                        "source": source_counts["fdb_setup_definition"],
                        "inserted": target_counts["fdb_setup"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "fdb_operation (from processing_run + analysis_run)": {
                        "source": source_counts["fdb_processing_run"] + source_counts["fdb_analysis_run"],
                        "inserted": target_counts["fdb_operation"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "fdb_operation_artifact (from processing_input + provenance_edge)": {
                        "source": source_counts["fdb_processing_input"] + source_counts["fdb_provenance_edge"],
                        "inserted": target_counts["fdb_operation_artifact"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "fdb_edge (from provenance_edge)": {
                        "source": source_counts["fdb_provenance_edge"],
                        "inserted": target_counts["fdb_edge"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "fdb_parameter (from analysis_parameter)": {
                        "source": source_counts["fdb_analysis_parameter"],
                        "inserted": target_counts["fdb_parameter"],
                        "skipped": 0,
                        "warnings": [],
                    },
                }
                logger.info("Migration report:\n%s", report.summary)

            # Phase 18: Rename canonical tables to mfdb_ prefix, add new tables
            if version < 18:
                logger.info("Migrating database schema to version 18 (rename fdb_ -> mfdb_ canonical tables)...")

                # 1. Create mfdb_schema_version
                cursor.execute("CREATE TABLE IF NOT EXISTS mfdb_schema_version (version INTEGER)")

                # 2. Create canonical mfdb_ tables. Do not create
                # mfdb_sample/mfdb_experiment here: flr_sample and
                # flr_experiment are the canonical sample/experiment domain
                # model for MFDB.
                for sql in _CANONICAL_CHECK_SQL + CREATE_TABLES_SQL:
                    if any(t in sql for t in [
                        "mfdb_artifact", "mfdb_operation", "mfdb_operation_artifact",
                        "mfdb_edge", "mfdb_parameter", "mfdb_setup", "mfdb_audit_log",
                        "mfdb_vocabulary"
                    ]):
                        cursor.execute(sql)

                # 3. Create canonical mfdb_ indices. Skip experimental
                # mfdb_sample/mfdb_experiment indices for the same reason.
                for sql in CREATE_INDICES_SQL:
                    if any(t in sql for t in [
                        "idx_mfdb_artifact", "idx_mfdb_operation", "idx_mfdb_edge",
                        "idx_mfdb_parameter", "idx_mfdb_setup", "idx_mfdb_audit_log",
                        "idx_mfdb_vocabulary"
                    ]):
                        cursor.execute(sql)

                # 4. Perform backfill
                conn.row_factory = sqlite3.Row
                current_tables = {
                    r[0] for r in cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }

                # 4.1 Backfill fdb_artifact -> mfdb_artifact
                if "fdb_artifact" in current_tables:
                    src_rows = conn.execute("SELECT * FROM fdb_artifact").fetchall()
                    for row in src_rows:
                        conn.execute(
                            """INSERT OR IGNORE INTO mfdb_artifact (
                                artifact_id, artifact_kind, data_format, experiment_id, storage_mode,
                                file_path, url, folder_path, mime_type, size_bytes, checksum,
                                checksum_algorithm, row_count, validation_status, validation_message,
                                metadata_json, data_json, data_blob, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                row["artifact_id"],
                                row["artifact_type"],
                                None,
                                row["experiment_id"],
                                row["storage_mode"],
                                row["file_path"],
                                row["url"],
                                row["folder_path"],
                                row["mime_type"],
                                row["size_bytes"],
                                row["checksum"],
                                row["checksum_algorithm"],
                                row["row_count"],
                                row["validation_status"],
                                row["validation_message"],
                                row["metadata_json"],
                                row["data_json"],
                                row["data_blob"],
                                row["created_at"],
                                row["updated_at"],
                            )
                        )

                # 4.2 Backfill fdb_operation -> mfdb_operation
                if "fdb_operation" in current_tables:
                    src_rows = conn.execute("SELECT * FROM fdb_operation").fetchall()
                    for row in src_rows:
                        conn.execute(
                            """INSERT OR IGNORE INTO mfdb_operation (
                                operation_id, operation_type, experiment_id, setup_id,
                                settings_json, settings_hash, operator_user_id,
                                software_package, software_module, software_version,
                                runtime_environment_json, started_at, ended_at, status,
                                error_message, traceback_summary, metadata_json,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                row["operation_id"],
                                row["operation_type"],
                                row["experiment_id"],
                                row["setup_id"],
                                row["settings_json"],
                                row["settings_hash"],
                                row["operator_user_id"],
                                row["software_package"],
                                row["software_module"],
                                row["software_version"],
                                row["runtime_environment_json"],
                                row["started_at"],
                                row["ended_at"],
                                row["status"],
                                row["error_message"],
                                row["traceback_summary"],
                                row["metadata_json"],
                                row["created_at"],
                                row["updated_at"],
                            )
                        )

                # 4.3 Backfill fdb_operation_artifact -> mfdb_operation_artifact
                if "fdb_operation_artifact" in current_tables:
                    src_rows = conn.execute("SELECT * FROM fdb_operation_artifact").fetchall()
                    for row in src_rows:
                        conn.execute(
                            """INSERT OR IGNORE INTO mfdb_operation_artifact (
                                operation_id, artifact_id, direction, role, ordinal,
                                checksum_snapshot, metadata_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                row["operation_id"],
                                row["artifact_id"],
                                row["direction"],
                                row["role"],
                                row["ordinal"],
                                row["checksum_snapshot"],
                                row["metadata_json"],
                            )
                        )

                # 4.4 Backfill fdb_edge -> mfdb_edge
                if "fdb_edge" in current_tables:
                    src_rows = conn.execute("SELECT * FROM fdb_edge").fetchall()
                    for row in src_rows:
                        if row["relationship_type"] in ("input_to", "produced"):
                            if (
                                row["relationship_type"] == "input_to"
                                and row["target_node_type"] in ("operation", "processing_run", "analysis_run")
                            ):
                                conn.execute(
                                    """INSERT OR IGNORE INTO mfdb_operation_artifact (
                                        operation_id, artifact_id, direction, role, metadata_json
                                    ) VALUES (?, ?, 'input', ?, ?)""",
                                    (
                                        row["target_node_id"], row["source_node_id"],
                                        row["source_node_type"], row["metadata_json"],
                                    ),
                                )
                            elif (
                                row["relationship_type"] == "produced"
                                and row["source_node_type"] in ("operation", "processing_run", "analysis_run")
                            ):
                                conn.execute(
                                    """INSERT OR IGNORE INTO mfdb_operation_artifact (
                                        operation_id, artifact_id, direction, role, metadata_json
                                    ) VALUES (?, ?, 'output', ?, ?)""",
                                    (
                                        row["source_node_id"], row["target_node_id"],
                                        row["target_node_type"], row["metadata_json"],
                                    ),
                                )
                            continue
                        conn.execute(
                            """INSERT OR IGNORE INTO mfdb_edge (
                                edge_id, source_node_type, source_node_id, target_node_type, target_node_id,
                                relationship_type, operation_id, settings_hash, timestamp, software_version,
                                checksum_snapshot_json, metadata_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                row["edge_id"],
                                row["source_node_type"],
                                row["source_node_id"],
                                row["target_node_type"],
                                row["target_node_id"],
                                row["relationship_type"],
                                row["operation_id"],
                                row["settings_hash"],
                                row["timestamp"],
                                row["software_version"],
                                row["checksum_snapshot_json"],
                                row["metadata_json"],
                            ),
                        )

                # 4.5 Backfill fdb_parameter -> mfdb_parameter
                if "fdb_parameter" in current_tables:
                    src_rows = conn.execute("SELECT * FROM fdb_parameter").fetchall()
                    for row in src_rows:
                        conn.execute(
                            """INSERT OR IGNORE INTO mfdb_parameter (
                                parameter_id, parameter_uuid, operation_id, name, value, standard_error,
                                confidence_interval_low, confidence_interval_high, initial_value,
                                lower_bound, upper_bound, bounds_on, units, parameter_type, expression,
                                prior_json, mapping_json, metadata_json, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                row["parameter_id"],
                                row["parameter_uuid"],
                                row["operation_id"],
                                row["name"],
                                row["value"],
                                row["standard_error"],
                                row["confidence_interval_low"],
                                row["confidence_interval_high"],
                                row["initial_value"],
                                row["lower_bound"],
                                row["upper_bound"],
                                row["bounds_on"],
                                row["units"],
                                row["parameter_type"],
                                row["expression"],
                                row["prior_json"],
                                row["mapping_json"],
                                row["metadata_json"],
                                row["created_at"],
                                row["updated_at"],
                            )
                        )

                # 4.6 Backfill fdb_setup -> mfdb_setup
                if "fdb_setup" in current_tables:
                    src_rows = conn.execute("SELECT * FROM fdb_setup").fetchall()
                    for row in src_rows:
                        conn.execute(
                            """INSERT OR IGNORE INTO mfdb_setup (
                                setup_id, name, version, instrument_id, description, configuration_json,
                                detectors_json, timing_calibration_json, irf_definition_json,
                                dark_count_json, timing_resolution_json, burst_defaults_json,
                                fcs_calibration_json, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                row["setup_id"],
                                row["name"],
                                row["version"],
                                row["instrument_id"],
                                row["description"],
                                row["configuration_json"],
                                row["detectors_json"],
                                row["timing_calibration_json"],
                                row["irf_definition_json"],
                                row["dark_count_json"],
                                row["timing_resolution_json"],
                                row["burst_defaults_json"],
                                row["fcs_calibration_json"],
                                row["created_at"],
                                row["updated_at"],
                            )
                        )

                # 4.7 Backfill fdb_audit_log -> mfdb_audit_log
                if "fdb_audit_log" in current_tables:
                    src_rows = conn.execute("SELECT * FROM fdb_audit_log").fetchall()
                    for row in src_rows:
                        conn.execute(
                            """INSERT OR IGNORE INTO mfdb_audit_log (
                                log_id, timestamp, action, target_type, target_id,
                                operator_user_id, details_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                row["log_id"],
                                row["timestamp"],
                                row["action"],
                                row["target_type"],
                                row["target_id"],
                                row["operator_user_id"],
                                row["details_json"],
                                row["created_at"],
                            )
                        )

                set_schema_version(conn, 18)
                version = 18
                logger.info("Successfully migrated schema and backfilled data to version 18.")

                # Build migration report
                report = MigrationReport(from_version=17, to_version=18)
                report.tables_added = [
                    "mfdb_artifact", "mfdb_operation", "mfdb_operation_artifact",
                    "mfdb_edge", "mfdb_parameter", "mfdb_setup", "mfdb_audit_log",
                    "mfdb_schema_version", "mfdb_vocabulary",
                ]

                mfdb_target_counts = {}
                for t in ["mfdb_artifact", "mfdb_operation", "mfdb_operation_artifact",
                          "mfdb_edge", "mfdb_parameter", "mfdb_setup", "mfdb_audit_log"]:
                    try:
                        mfdb_target_counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    except Exception:
                        mfdb_target_counts[t] = 0

                fdb_source_counts = {}
                for t in ["fdb_artifact", "fdb_operation", "fdb_operation_artifact",
                          "fdb_edge", "fdb_parameter", "fdb_setup", "fdb_audit_log"]:
                    try:
                        fdb_source_counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    except Exception:
                        fdb_source_counts[t] = 0

                report.backfill = {
                    "mfdb_artifact": {
                        "source": fdb_source_counts["fdb_artifact"],
                        "inserted": mfdb_target_counts["mfdb_artifact"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "mfdb_operation": {
                        "source": fdb_source_counts["fdb_operation"],
                        "inserted": mfdb_target_counts["mfdb_operation"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "mfdb_operation_artifact": {
                        "source": fdb_source_counts["fdb_operation_artifact"],
                        "inserted": mfdb_target_counts["mfdb_operation_artifact"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "mfdb_edge": {
                        "source": fdb_source_counts["fdb_edge"],
                        "inserted": mfdb_target_counts["mfdb_edge"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "mfdb_parameter": {
                        "source": fdb_source_counts["fdb_parameter"],
                        "inserted": mfdb_target_counts["mfdb_parameter"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "mfdb_setup": {
                        "source": fdb_source_counts["fdb_setup"],
                        "inserted": mfdb_target_counts["mfdb_setup"],
                        "skipped": 0,
                        "warnings": [],
                    },
                    "mfdb_audit_log": {
                        "source": fdb_source_counts["fdb_audit_log"],
                        "inserted": mfdb_target_counts["mfdb_audit_log"],
                        "skipped": 0,
                        "warnings": [],
                    },
                }
                logger.info("Migration report:\n%s", report.summary)

            if version < 19:
                logger.info("Migrating database schema to version 19 (enforce mfdb_edge relationship invariant)...")
                from_version = version
                edge_report = _ensure_mfdb_edge_constraint(conn)
                set_schema_version(conn, 19)
                version = 19
                report = MigrationReport(
                    from_version=from_version,
                    to_version=19,
                    tables_added=["mfdb_edge.relationship_type CHECK constraint"],
                    backfill={
                        "mfdb_edge operation relationships": {
                            "source": edge_report.get("converted", 0) + edge_report.get("skipped", 0),
                            "inserted": edge_report.get("converted", 0),
                            "skipped": edge_report.get("skipped", 0),
                            "warnings": edge_report.get("warnings", []),
                        },
                    },
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 20:
                logger.info("Migrating database schema to version 20 (enforce mfdb_edge vocabulary triggers)...")
                from_version = version
                edge_report = _ensure_mfdb_edge_vocabulary_triggers(conn)
                set_schema_version(conn, 20)
                version = 20
                report = MigrationReport(
                    from_version=from_version,
                    to_version=20,
                    tables_added=["mfdb_edge.relationship_type active-vocabulary triggers"],
                    backfill={
                        "mfdb_edge relationship vocabulary": {
                            "source": edge_report.get("source", 0),
                            "inserted": edge_report.get("inserted", 0),
                            "skipped": edge_report.get("skipped", 0),
                            "warnings": edge_report.get("warnings", []),
                        },
                    },
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 21:
                logger.info("Migrating database schema to version 21 (enhance user data fields)...")
                import uuid
                from_version = version
                # Check for existing columns and alter table
                for col in ["user_uuid", "department", "role", "address", "website", "phone"]:
                    try:
                        cursor.execute(f"ALTER TABLE flr_sample_users ADD COLUMN {col} TEXT")
                    except sqlite3.OperationalError:
                        pass

                # Ensure unique constraint on user_uuid by creating a unique index if SQLite supports it,
                # but to keep it simple and compliant with ALTER TABLE restrictions, we just rely on standard columns
                # and UNIQUE index.
                try:
                    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_flr_sample_users_uuid ON flr_sample_users(user_uuid)")
                except sqlite3.OperationalError:
                    pass

                # Backfill UUIDs for users that don't have them
                rows = cursor.execute("SELECT user_id, user_uuid FROM flr_sample_users").fetchall()
                updated_count = 0
                for user_id, user_uuid in rows:
                    if not user_uuid:
                        new_uuid = str(uuid.uuid4())
                        cursor.execute("UPDATE flr_sample_users SET user_uuid = ? WHERE user_id = ?", (new_uuid, user_id))
                        updated_count += 1

                set_schema_version(conn, 21)
                version = 21
                report = MigrationReport(
                    from_version=from_version,
                    to_version=21,
                    tables_added=["flr_sample_users new columns"],
                    backfill={
                        "users uuid backfilled": {
                            "source": len(rows),
                            "inserted": updated_count,
                            "skipped": len(rows) - updated_count,
                            "warnings": [],
                        },
                    },
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 22:
                logger.info("Migrating database schema to version 22 (add user passwords and admin privileges)...")
                from_version = version
                for col, col_type in [("is_admin", "INTEGER DEFAULT 0"), ("password_hash", "TEXT")]:
                    try:
                        cursor.execute(f"ALTER TABLE flr_sample_users ADD COLUMN {col} {col_type}")
                    except sqlite3.OperationalError:
                        pass

                set_schema_version(conn, 22)
                version = 22
                report = MigrationReport(
                    from_version=from_version,
                    to_version=22,
                    tables_added=["flr_sample_users password and admin columns"],
                    backfill={},
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 23:
                logger.info(
                    "Migrating database schema to version 23 "
                    "(add lifecycle columns created_at, updated_at, deleted_at)..."
                )
                from_version = version
                now = _utc_now()

                # Tables grouped by which lifecycle columns they already have
                # (existing_created, existing_updated) -> columns to add
                lifecycle_table_config: list[tuple[str, bool, bool]] = [
                    # Tables with both created_at and updated_at — only need deleted_at
                    ("fdb_raw_data", True, True),
                    ("fdb_processing_run", True, True),
                    ("fdb_processed_data", True, True),
                    ("fdb_setup_definition", True, True),
                    ("fdb_analysis_run", True, True),
                    ("fdb_analysis_parameter", True, True),
                    ("fdb_setup", True, True),
                    ("fdb_artifact", True, True),
                    ("fdb_operation", True, True),
                    ("fdb_parameter", True, True),
                    ("mfdb_artifact", True, True),
                    ("mfdb_operation", True, True),
                    ("mfdb_parameter", True, True),
                    ("mfdb_setup", True, True),
                    ("mfdb_sample", True, True),
                    ("mfdb_experiment", True, True),
                    # Tables with only created_at — need updated_at and deleted_at
                    ("fdb_audit_log", True, False),
                    ("mfdb_audit_log", True, False),
                    # Tables with no lifecycle columns — need all three
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
                    ("fdb_processing_input", False, False),
                    ("fdb_provenance_edge", False, False),
                    ("fdb_operation_artifact", False, False),
                    ("fdb_edge", False, False),
                    ("mfdb_operation_artifact", False, False),
                    ("mfdb_edge", False, False),
                    ("mfdb_vocabulary", False, False),
                ]

                for table, has_created, has_updated in lifecycle_table_config:
                    try:
                        _ensure_column(conn, table, "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
                        _ensure_column(conn, table, "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
                        _ensure_column(conn, table, "deleted_at", "TEXT")

                        # Backfill: set created_at = now for rows with NULL
                        conn.execute(
                            f"UPDATE {table} SET created_at = ? WHERE created_at IS NULL",
                            (now,),
                        )

                        # Backfill: set updated_at = now for rows with NULL
                        if not has_updated:
                            conn.execute(
                                f"UPDATE {table} SET updated_at = ? WHERE updated_at IS NULL",
                                (now,),
                            )

                        # deleted_at stays NULL for active rows (default)
                    except sqlite3.OperationalError:
                        pass

                set_schema_version(conn, 23)
                version = 23
                report = MigrationReport(
                    from_version=from_version,
                    to_version=23,
                    tables_added=[
                        "lifecycle columns (created_at, updated_at, deleted_at) "
                        "on all mfdb_*, flr_*, and fdb_* data tables",
                    ],
                    backfill={
                        "lifecycle columns": {
                            "source": sum(
                                _safe_table_count(conn, t[0])
                                for t in lifecycle_table_config
                            ),
                            "inserted": 0,
                            "skipped": 0,
                            "warnings": [],
                        },
                    },
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 24:
                logger.info(
                    "Migrating database schema to version 24 "
                    "(add mfdb_branch table and active_branch_uuid to flr_sample_users)..."
                )
                from_version = version

                for sql in CREATE_TABLES_SQL:
                    cursor.execute(sql)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mfdb_branch (
                        branch_uuid TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        parent_branch_uuid TEXT REFERENCES mfdb_branch(branch_uuid),
                        head_operation_id TEXT REFERENCES mfdb_operation(operation_id),
                        created_by_user_id TEXT REFERENCES flr_sample_users(user_id),
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        deleted_at TEXT
                    )
                """)

                try:
                    cursor.execute("ALTER TABLE flr_sample_users ADD COLUMN active_branch_uuid TEXT")
                except sqlite3.OperationalError:
                    pass

                import uuid
                main_uuid = "00000000-0000-0000-0000-000000000000"
                existing_main = cursor.execute("SELECT branch_uuid FROM mfdb_branch WHERE name = 'main'").fetchone()
                if not existing_main:
                    cursor.execute(
                        "INSERT INTO mfdb_branch (branch_uuid, name, description) VALUES (?, ?, ?)",
                        (main_uuid, "main", "Default main branch")
                    )
                else:
                    main_uuid = existing_main[0]

                cursor.execute("UPDATE flr_sample_users SET active_branch_uuid = ? WHERE active_branch_uuid IS NULL", (main_uuid,))

                set_schema_version(conn, 24)
                version = 24
                report = MigrationReport(
                    from_version=from_version,
                    to_version=24,
                    tables_added=["mfdb_branch"],
                    backfill={"mfdb_branch": {"source": 1, "inserted": 1, "skipped": 0}},
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 25:
                logger.info(
                    "Migrating database schema to version 25 "
                    "(add auth tables: groups, ACLs, sessions, auth_attempt)..."
                )
                from_version = version

                for sql in CREATE_TABLES_SQL:
                    if any(t in sql for t in [
                        "mfdb_group",
                        "mfdb_group_member",
                        "mfdb_object_acl",
                        "mfdb_acl_entry",
                        "mfdb_session",
                        "mfdb_auth_attempt",
                    ]):
                        cursor.execute(sql)

                for sql in CREATE_INDICES_SQL:
                    if any(t in sql for t in [
                        "idx_mfdb_group_member",
                        "idx_mfdb_object_acl",
                        "idx_mfdb_acl_entry",
                        "idx_mfdb_session",
                        "idx_mfdb_auth_attempt",
                    ]):
                        cursor.execute(sql)

                bootstrap_auth_groups(conn)

                set_schema_version(conn, 25)
                version = 25
                report = MigrationReport(
                    from_version=from_version,
                    to_version=25,
                    tables_added=[
                        "mfdb_group", "mfdb_group_member",
                        "mfdb_object_acl", "mfdb_acl_entry",
                        "mfdb_session", "mfdb_auth_attempt",
                    ],
                    backfill={
                        "auth groups": {"source": 0, "inserted": 0, "skipped": 0},
                    },
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 26:
                logger.info(
                    "Migrating database schema to version 26 "
                    "(add allow_passwordless_login column)..."
                )
                from_version = version
                try:
                    cursor.execute(
                        "ALTER TABLE flr_sample_users ADD COLUMN allow_passwordless_login INTEGER DEFAULT 0"
                    )
                except sqlite3.OperationalError:
                    pass  # column may already exist
                set_schema_version(conn, 26)
                version = 26
                report = MigrationReport(
                    from_version=from_version,
                    to_version=26,
                    tables_added=[],
                )
                logger.info("Migration report:\n%s", report.summary)

            if version < 27:
                logger.info(
                    "Migrating database schema to version 27 "
                    "(add mfdb_object table and object_uuid to mfdb_artifact)..."
                )
                from_version = version

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mfdb_object (
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
                    )
                """)

                try:
                    cursor.execute(
                        "ALTER TABLE mfdb_artifact ADD COLUMN object_uuid TEXT REFERENCES mfdb_object(object_uuid)"
                    )
                except sqlite3.OperationalError:
                    pass

                set_schema_version(conn, 27)
                version = 27
                report = MigrationReport(
                    from_version=from_version,
                    to_version=27,
                    tables_added=["mfdb_object"],
                    backfill={},
                )
                logger.info("Migration report:\n%s", report.summary)

            # --- v28: fix mfdb_operation_artifact PK to include role ---
            if version < 28:
                _fix_operation_artifact_pk(conn)
                set_schema_version(conn, 28)
                version = 28

            _ensure_lifecycle_columns(conn)

        finally:
            cursor.close()
    bootstrap_vocabulary(conn)
    # Ensure auth columns exist on flr_sample_users (for DBs that skipped v22 migration)
    for col, col_type in [("is_admin", "INTEGER DEFAULT 0"), ("password_hash", "TEXT"), ("allow_passwordless_login", "INTEGER DEFAULT 0")]:
        try:
            with conn:
                conn.execute(f"ALTER TABLE flr_sample_users ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass
    bootstrap_default_user(conn)
    try:
        bootstrap_auth_groups(conn)
    except sqlite3.OperationalError:
        pass
    return report
