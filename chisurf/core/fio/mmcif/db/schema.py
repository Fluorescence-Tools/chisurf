import sqlite3
import logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 6

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

    "CREATE TABLE IF NOT EXISTS flr_sample (sample_id TEXT PRIMARY KEY, description TEXT, details TEXT)",

    """CREATE TABLE IF NOT EXISTS flr_sample_condition (
        condition_id TEXT PRIMARY KEY,
        ph REAL,
        temperature REAL,
        ionic_strength REAL,
        buffer_composition TEXT,
        details TEXT
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
        UNIQUE (probe_id, property_name)
    )""",
    
    """CREATE TABLE IF NOT EXISTS spectra (
        id INTEGER PRIMARY KEY,
        probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
        spectrum_type TEXT NOT NULL,
        wavelengths BLOB NOT NULL,
        intensity_values BLOB NOT NULL,
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
        sample_id TEXT REFERENCES flr_sample(sample_id),
        method TEXT,
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
        details TEXT
    )"""
]

CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_probes_type ON probes (type_id)",
    "CREATE INDEX IF NOT EXISTS idx_probes_cat ON probes (category)",
    "CREATE INDEX IF NOT EXISTS idx_op_probe ON optical_properties (probe_id)",
    "CREATE INDEX IF NOT EXISTS idx_spectra_probe ON spectra (probe_id)"
]

def migrate_schema(conn: sqlite3.Connection):
    """Execution of versioned migration logic."""
    with conn:
        cursor = conn.cursor()
        try:
            existing = {r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

            if not existing:
                for sql in CREATE_TABLES_SQL + CREATE_INDICES_SQL:
                    cursor.execute(sql)
                set_schema_version(conn, SCHEMA_VERSION)
                return

            version = get_schema_version(conn)

            # Phase 1: Legacy -> v1 (Basic flrCIF Renames)
            if version < 1:
                if "item_types" in existing and "probe_types" not in existing:
                    cursor.execute("ALTER TABLE item_types RENAME TO probe_types")
                    tp_cols = {r[1] for r in cursor.execute("PRAGMA table_info(probe_types)").fetchall()}
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
                    "description": "TEXT DEFAULT ''"
                }
                for col, definition in to_add.items():
                    if col not in cols:
                        cursor.execute(f"ALTER TABLE probes ADD COLUMN {col} {definition}")

                for table in ["spectra", "optical_properties"]:
                    if table in existing:
                        t_cols = {r[1] for r in cursor.execute(f"PRAGMA table_info({table})").fetchall()}
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
                tp_cols = {r[1] for r in cursor.execute("PRAGMA table_info(probe_types)").fetchall()}
                if "id" in tp_cols and "type_id" not in tp_cols:
                    cursor.execute("ALTER TABLE probe_types RENAME COLUMN id TO type_id")
                if "name" in tp_cols and "type_name" not in tp_cols:
                    cursor.execute("ALTER TABLE probe_types RENAME COLUMN name TO type_name")

                # Fix probes renames and redundancy
                p_cols = {r[1] for r in cursor.execute("PRAGMA table_info(probes)").fetchall()}
                if "id" in p_cols and "probe_id" not in p_cols:
                    cursor.execute("ALTER TABLE probes RENAME COLUMN id TO probe_id")
                
                if "name" in p_cols and "chromophore_name" in p_cols:
                    cursor.execute("UPDATE probes SET chromophore_name = name WHERE chromophore_name IS NULL OR chromophore_name = ''")
                elif "name" in p_cols and "chromophore_name" not in p_cols:
                    cursor.execute("ALTER TABLE probes RENAME COLUMN name TO chromophore_name")

                set_schema_version(conn, 5)
                version = 5

            # Phase 6: Ensure probe_id column exists in related tables (v5 -> v6)
            if version < 6:
                for table in ["images", "spectra", "optical_properties"]:
                    if table in existing:
                        t_cols = {r[1] for r in cursor.execute(f"PRAGMA table_info({table})").fetchall()}
                        if "item_id" in t_cols and "probe_id" not in t_cols:
                            cursor.execute(f"ALTER TABLE {table} ADD COLUMN probe_id INTEGER")
                            cursor.execute(f"UPDATE {table} SET probe_id = item_id")
                set_schema_version(conn, 6)
                version = 6
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
