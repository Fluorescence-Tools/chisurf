import sqlite3
import logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 16

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
        type_id INTEGER REFERENCES probe_types(type_id)
    )""",
    """CREATE TABLE IF NOT EXISTS entities (
        entity_id TEXT PRIMARY KEY,
        type TEXT DEFAULT 'polymer',
        description TEXT,
        formula_weight REAL,
        src_method TEXT,
        number_of_molecules INTEGER DEFAULT 1,
        common_name TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS entity_poly_seq (
        id INTEGER PRIMARY KEY,
        entity_id TEXT NOT NULL REFERENCES entities(entity_id),
        num INTEGER NOT NULL,
        mon_id TEXT NOT NULL,
        hetero TEXT DEFAULT 'n',
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
        sample_uuid TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_condition (
        condition_id TEXT PRIMARY KEY,
        ph REAL,
        temperature REAL,
        ionic_strength REAL,
        buffer_composition TEXT,
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_users (
        user_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        email TEXT,
        affiliation TEXT,
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_devices (
        device_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        device_type TEXT,
        model TEXT,
        serial_number TEXT,
        location TEXT,
        owner TEXT,
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_experiment_type (
        type_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        category TEXT,
        description TEXT,
        details TEXT
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
        setup_definition_id TEXT REFERENCES fdb_setup_definition(setup_id) ON DELETE SET NULL
    )""",
    """CREATE TABLE IF NOT EXISTS flr_experiment_key_value (
        experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT,
        details TEXT,
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
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_key_value (
        sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT,
        details TEXT,
        UNIQUE (sample_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_sample_probe (
        sample_probe_id INTEGER PRIMARY KEY,
        sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        poly_probe_position_id INTEGER REFERENCES flr_poly_probe_position(id),
        fluorophore_type TEXT DEFAULT 'unspecified',
        description TEXT,
        UNIQUE (sample_id, probe_id, poly_probe_position_id)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_poly_probe_position (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        entity_id TEXT NOT NULL REFERENCES entities(entity_id),
        asym_id TEXT DEFAULT 'A',
        residue_number INTEGER NOT NULL,
        residue_name TEXT,
        description TEXT,
        UNIQUE (probe_id, entity_id, asym_id, residue_number)
    )""",
    """CREATE TABLE IF NOT EXISTS optical_properties (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        property_name TEXT NOT NULL,
        property_value TEXT,
        unit TEXT,
        details TEXT,
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
        UNIQUE (probe_id, spectrum_type)
    )""",
    """CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        image_name TEXT,
        image_data BLOB,
        image_format TEXT
    )""",
    "CREATE TABLE IF NOT EXISTS flr_instrument (instrument_id TEXT PRIMARY KEY, instrument_name TEXT, details TEXT)",
    """CREATE TABLE IF NOT EXISTS flr_inst_setting (
        id INTEGER PRIMARY KEY,
        instrument_id TEXT NOT NULL REFERENCES flr_instrument(instrument_id),
        setting_name TEXT NOT NULL,
        setting_value TEXT,
        details TEXT
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
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_fret_calibration_parameters (
        id INTEGER PRIMARY KEY,
        analysis_id TEXT NOT NULL REFERENCES flr_fret_analysis(analysis_id),
        phi_acceptor REAL,
        alpha REAL,
        gamma REAL,
        delta REAL,
        beta REAL,
        details TEXT
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
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_entity_assembly (
        assembly_id TEXT PRIMARY KEY,
        description TEXT,
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_exp_condition (
        condition_id TEXT PRIMARY KEY,
        details TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS flr_experiment (
        id TEXT PRIMARY KEY,
        ordinal_id INTEGER,
        instrument_id TEXT,
        inst_setting_id TEXT,
        exp_condition_id TEXT,
        sample_id TEXT,
        details TEXT,
        setup_definition_id TEXT REFERENCES fdb_setup_definition(setup_id) ON DELETE SET NULL
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
        UNIQUE (analysis_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS flr_photon_stream (
        stream_id TEXT PRIMARY KEY,
        analysis_id TEXT,
        external_file_id INTEGER REFERENCES ihm_external_files(id),
        detector_id TEXT,
        description TEXT,
        details TEXT
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
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_processing_input (
        processing_id TEXT NOT NULL REFERENCES fdb_processing_run(processing_id) ON DELETE CASCADE,
        raw_data_id TEXT NOT NULL REFERENCES fdb_raw_data(raw_data_id) ON DELETE CASCADE,
        ordinal INTEGER DEFAULT 0,
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
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
        metadata_json TEXT
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
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS fdb_audit_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        action TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        operator_user_id TEXT,
        details_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
]

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
]


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add a column to an existing SQLite table when it is missing.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    table : str
        Table name.
    column : str
        Column name.
    definition : str
        SQLite column definition.
    """
    try:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.OperationalError:
        return
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _ensure_foreign_key(conn: sqlite3.Connection, table: str, column: str, parent: str) -> None:
    """Add a simple foreign-key column to legacy SQLite tables when missing.

    SQLite cannot add a foreign-key constraint with ALTER TABLE, so this helper
    adds the nullable column. New databases created with CREATE_TABLES_SQL get
    the full constraint.
    """
    _ensure_column(conn, table, column, "INTEGER")
    try:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table} ({column})")
    except sqlite3.OperationalError:
        pass


def migrate_schema(conn: sqlite3.Connection):
    """Execution of versioned migration logic."""
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
                for sql in CREATE_TABLES_SQL + CREATE_INDICES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, SCHEMA_VERSION)
                return

            version = get_schema_version(conn)

            if "flr_experiment" in existing:
                t_cols = {
                    r[1] for r in cursor.execute("PRAGMA table_info(flr_experiment)").fetchall()
                }
                if "id" in t_cols and "experiment_id" not in t_cols:
                    cursor.execute("ALTER TABLE flr_experiment RENAME COLUMN id TO experiment_id")

            # Phase 1: Legacy -> v1 (Basic flrCIF Renames)
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

            # Phase 2: Experimental Tables (v1 -> v2)
            if version < 2:
                for sql in CREATE_TABLES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, 2)
                version = 2

            # Phase 3: Indices & Optimization (v2 -> v3)
            if version < 3:
                for sql in CREATE_INDICES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, 3)
                version = 3

            # Phase 4: Data Normalization (v3 -> v4)
            if version < 4:
                set_schema_version(conn, 4)
                version = 4

            # Phase 5: Robust Rename Fixes (v4 -> v5)
            if version < 5:
                # Fix probe_types renames that might have been skipped
                tp_cols = {
                    r[1] for r in cursor.execute("PRAGMA table_info(probe_types)").fetchall()
                }
                if "id" in tp_cols and "type_id" not in tp_cols:
                    cursor.execute("ALTER TABLE probe_types RENAME COLUMN id TO type_id")
                if "name" in tp_cols and "type_name" not in tp_cols:
                    cursor.execute("ALTER TABLE probe_types RENAME COLUMN name TO type_name")

                # Fix probes renames and redundancy
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

            # Phase 6: Ensure probe_id column exists in related tables (v5 -> v6)
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

            # Phase 7: Official-ish FLR metadata, external files, and spectra units (v6 -> v7)
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
                _ensure_column(
                    cursor, "flr_poly_probe_position", "mutation_flag", "TEXT DEFAULT 'no'"
                )
                _ensure_column(
                    cursor, "flr_poly_probe_position", "modification_flag", "TEXT DEFAULT 'no'"
                )
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
                _ensure_column(
                    cursor, "flr_fret_distance_restraint", "peak_assignment_id", "INTEGER"
                )

                cursor.execute(
                    "UPDATE spectra SET wavelength_unit = 'nm' WHERE wavelength_unit IS NULL"
                )
                cursor.execute(
                    "UPDATE spectra SET intensity_unit = 'normalized' WHERE intensity_unit IS NULL"
                )
                cursor.execute(
                    "UPDATE flr_poly_probe_position SET mutation_flag = 'no' WHERE mutation_flag IS NULL"
                )
                cursor.execute(
                    "UPDATE flr_poly_probe_position SET modification_flag = 'no' WHERE modification_flag IS NULL"
                )
                cursor.execute(
                    "UPDATE flr_fret_forster_radius "
                    "SET reduced_forster_radius = forster_radius "
                    "WHERE reduced_forster_radius IS NULL AND forster_radius IS NOT NULL"
                )

                for sql in CREATE_INDICES_SQL:
                    cursor.execute(sql)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_analysis_metadata_analysis ON analysis_metadata (analysis_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_photon_stream_analysis ON flr_photon_stream (analysis_id)"
                )
                _ensure_column(cursor, "ihm_external_files", "md5", "TEXT")
                _ensure_column(cursor, "ihm_external_files", "uuid", "TEXT")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_external_files_path ON ihm_external_files (file_path)"
                )

                set_schema_version(conn, 7)
                version = 7

            # Phase 8: Sample UUID (v7 -> v8)
            if version < 8:
                _ensure_column(cursor, "flr_sample", "sample_uuid", "TEXT")
                cursor.execute(
                    "UPDATE flr_sample SET sample_uuid = hex(randomblob(16)) WHERE sample_uuid IS NULL"
                )
                set_schema_version(conn, 8)
                version = 8

            # Phase 9: Explicit sample-probe mapping (v8 -> v9)
            if version < 9:
                for sql in [
                    "CREATE TABLE IF NOT EXISTS flr_sample_probe (sample_probe_id INTEGER PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE, probe_id INTEGER NOT NULL REFERENCES probes(probe_id), poly_probe_position_id INTEGER REFERENCES flr_poly_probe_position(id), fluorophore_type TEXT DEFAULT 'unspecified', description TEXT, UNIQUE (sample_id, probe_id, poly_probe_position_id))"
                ]:
                    cursor.execute(sql)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sample_probe_sample ON flr_sample_probe (sample_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sample_probe_probe ON flr_sample_probe (probe_id)"
                )
                set_schema_version(conn, 9)
                version = 9

            # Phase 10: Embeddable small analysis data (v9 -> v10)
            if version < 10:
                for sql in [
                    "CREATE TABLE IF NOT EXISTS analysis_data (id INTEGER PRIMARY KEY, analysis_id TEXT NOT NULL REFERENCES flr_fret_analysis(analysis_id), data_type TEXT NOT NULL, data_name TEXT, x_values BLOB NOT NULL, y_values BLOB NOT NULL, x_unit TEXT, y_unit TEXT, details TEXT)"
                ]:
                    cursor.execute(sql)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_analysis_data_analysis ON analysis_data (analysis_id)"
                )
                set_schema_version(conn, 10)
                version = 10

            # Phase 11: LIMS sample metadata, users, and devices (v10 -> v11)
            if version < 11:
                for table in ["flr_sample"]:
                    if table in existing:
                        for column, definition in {
                            "project_id": "TEXT",
                            "measured_by_user_id": "TEXT",
                            "measured_by_device_id": "TEXT",
                            "measured_at": "TEXT",
                        }.items():
                            _ensure_column(cursor, table, column, definition)
                for sql in [
                    "CREATE TABLE IF NOT EXISTS flr_sample_users (user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, email TEXT, affiliation TEXT, details TEXT)",
                    "CREATE TABLE IF NOT EXISTS flr_sample_devices (device_id TEXT PRIMARY KEY, name TEXT NOT NULL, device_type TEXT, model TEXT, serial_number TEXT, location TEXT, owner TEXT, details TEXT)",
                    "CREATE TABLE IF NOT EXISTS flr_sample_key_value (sample_id TEXT NOT NULL REFERENCES flr_sample(sample_id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT, details TEXT, UNIQUE (sample_id, key))",
                ]:
                    cursor.execute(sql)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sample_key_value_sample ON flr_sample_key_value (sample_id)"
                )
                set_schema_version(conn, 11)
                version = 11

            # Phase 12: Experiment LIMS records and data links (v11 -> v12)
            if version < 12:
                for table in ["flr_experiment"]:
                    if table in existing:
                        t_cols = {
                            r[1] for r in cursor.execute(f"PRAGMA table_info({table})").fetchall()
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
                            _ensure_column(cursor, table, column, definition)
                for sql in [
                    "CREATE TABLE IF NOT EXISTS flr_experiment_type (type_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, category TEXT, description TEXT, details TEXT)",
                    "CREATE TABLE IF NOT EXISTS flr_experiment_key_value (experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT, details TEXT, UNIQUE (experiment_id, key))",
                    "CREATE TABLE IF NOT EXISTS flr_experiment_data (data_id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES flr_experiment(experiment_id) ON DELETE CASCADE, data_type TEXT NOT NULL, storage_mode TEXT NOT NULL, file_path TEXT, url TEXT, folder_path TEXT, mime_type TEXT, size_bytes INTEGER, checksum TEXT, data_json TEXT, data_blob BLOB, reading_options_json TEXT, details TEXT)",
                ]:
                    cursor.execute(sql)
                for sql in CREATE_INDICES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, 12)
                version = 12

            # Phase 13: fdb burst TTTR provenance records (v12 -> v13)
            if version < 13:
                for sql in CREATE_TABLES_SQL:
                    if "fdb_" in sql:
                        cursor.execute(sql)
                for sql in CREATE_INDICES_SQL:
                    if "fdb_" in sql:
                        cursor.execute(sql)
                set_schema_version(conn, 13)
                version = 13

            # Phase 14: fdb setup definitions and linkage (v13 -> v14)
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

            # Phase 15: fdb analysis, model, parameter provenance (v14 -> v15)
            if version < 15:
                for sql in CREATE_TABLES_SQL:
                    if "fdb_analysis_run" in sql or "fdb_analysis_parameter" in sql:
                        cursor.execute(sql)
                for sql in CREATE_INDICES_SQL:
                    if "idx_fdb_analysis_run" in sql or "idx_fdb_analysis_parameter" in sql:
                        cursor.execute(sql)
                set_schema_version(conn, 15)
                version = 15

            # Phase 16: fdb audit logging (v15 -> v16)
            if version < 16:
                for sql in CREATE_TABLES_SQL:
                    if "fdb_audit_log" in sql:
                        cursor.execute(sql)
                for sql in CREATE_INDICES_SQL:
                    if "idx_fdb_audit_log" in sql:
                        cursor.execute(sql)
                set_schema_version(conn, 16)
                version = 16
        finally:
            cursor.close()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version of the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    int
        Schema version number, or 0 if not available.
    """
    try:
        row = conn.execute("SELECT version FROM _schema_version").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def set_schema_version(conn: sqlite3.Connection, version: int):
    """Set the schema version of the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    version : int
        Version number to set.
    """
    conn.execute("DELETE FROM _schema_version")
    conn.execute("INSERT INTO _schema_version (version) VALUES (?)", (version,))
