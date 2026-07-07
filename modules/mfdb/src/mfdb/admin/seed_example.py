"""Seed the MFDB with example single-molecule FRET data.

Populates the database with a demo DNA sample labeled with Alexa488/Alexa647,
a user (John Doe), an instrument (MT200), a smFRET experiment, and registers
actual SPC test data from the burst selection plugin with a burst-selection
processing run.

Usage::

    python -c "from mfdb.admin.seed_example import seed_example; seed_example()"
"""

from __future__ import annotations

import logging
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

from mfdb.store.database_resolver import resolve_database_path
from mfdb.repository import MFDatabase
from mfdb.samples.sample_manager import link_artifact_to_sample

logger = logging.getLogger(__name__)

SPC_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "burst"
    / "burst_selection"
    / "tests"
    / "data"
    / "bh_spc132_sm_dna"
)

DEMO = dict(
    user_id="john_doe",
    device_id="mt200",
    entity_id="sm_dna_demo",
    assembly_id="sm_dna_a488_a647",
    condition_id="pbs_ph74_25c",
    sample_id="sm_dna_a488_a647_sample",
    experiment_id="sm_dna_mt200_001",
    project_id="demo_sm_fret",
    processing_id="burst_selection_sm_dna_001",
)

DNA_SEQ = list("GAAGTCGAGATGGCTCGAGA")


def _spc_path(index: int = 0) -> Path:
    return SPC_DATA_DIR / f"m{index:03d}.spc"


def _upsert_probe(
    db: MFDatabase, name: str, type_id: int, *, category: str,
    abs_max: float, em_max: float, qy: float, ext_coeff: float | None = None,
) -> int:
    row = db.conn.execute(
        "SELECT probe_id FROM probes WHERE chromophore_name = ? AND type_id = ?",
        (name, type_id),
    ).fetchone()
    if row:
        return int(row["probe_id"])
    probe_id = db.add_probe(name, type_id, category=category)
    db.add_optical_property(probe_id, "abs_max", abs_max, unit="nm")
    db.add_optical_property(probe_id, "em_max", em_max, unit="nm")
    db.add_optical_property(probe_id, "qy", qy, unit="")
    if ext_coeff is not None:
        db.add_optical_property(probe_id, "ext_coeff", ext_coeff, unit="M-1 cm-1")
    logger.info("Created probe %s (id=%d)", name, probe_id)
    return probe_id


def _upsert_user(db: MFDatabase, user_id: str, display_name: str, email: str | None = None) -> None:
    if db.conn.execute("SELECT 1 FROM flr_sample_users WHERE user_id=?", (user_id,)).fetchone():
        return
    db.add_user(user_id, display_name, email=email)
    logger.info("Created user %s (%s)", user_id, display_name)


def _upsert_device(
    db: MFDatabase, device_id: str, name: str, /,
    device_type: str | None = None, model: str | None = None, serial: str | None = None,
) -> None:
    if db.conn.execute("SELECT 1 FROM flr_sample_devices WHERE device_id=?", (device_id,)).fetchone():
        return
    db.add_device(device_id, name, device_type=device_type, model=model, serial_number=serial)
    logger.info("Created device %s (%s)", device_id, name)


def _upsert_entity(db: MFDatabase, entity_id: str, seq: list[str]) -> None:
    """Insert directly -- add_entity API is incompatible with the schema."""
    if db.conn.execute("SELECT 1 FROM entities WHERE entity_id=?", (entity_id,)).fetchone():
        return
    with db.conn:
        db.conn.execute(
            "INSERT INTO entities (entity_id, type, description, common_name) "
            "VALUES (?, ?, ?, ?)",
            (entity_id, "polymer",
             "Demo DNA smFRET sample labeled with Alexa488/Alexa647", "Demo DNA"),
        )
        for i, mon_id in enumerate(seq, start=1):
            db.conn.execute(
                "INSERT INTO entity_poly_seq (entity_id, num, mon_id) VALUES (?, ?, ?)",
                (entity_id, i, mon_id),
            )
    logger.info("Created entity %s (%d bp)", entity_id, len(seq))


def _upsert_entity_assembly(db: MFDatabase, assembly_id: str, description: str) -> None:
    if db.conn.execute(
        "SELECT 1 FROM flr_entity_assembly WHERE assembly_id=?", (assembly_id,)
    ).fetchone():
        return
    with db.conn:
        db.conn.execute(
            "INSERT INTO flr_entity_assembly (assembly_id, description) VALUES (?, ?)",
            (assembly_id, description),
        )


def _upsert_sample_condition(
    db: MFDatabase, condition_id: str, *,
    ph: float, temperature: float, buffer: str,
) -> None:
    if db.conn.execute(
        "SELECT 1 FROM flr_sample_condition WHERE condition_id=?", (condition_id,)
    ).fetchone():
        return
    with db.conn:
        db.conn.execute(
            "INSERT INTO flr_sample_condition "
            "(condition_id, ph, temperature, buffer_composition, details) "
            "VALUES (?, ?, ?, ?, ?)",
            (condition_id, ph, temperature, buffer,
             f"pH {ph} at {temperature} K"),
        )


def _upsert_poly_probe_position(
    db: MFDatabase, probe_id: int, entity_id: str, residue: int, chain: str = "A",
) -> int:
    row = db.conn.execute(
        "SELECT id FROM flr_poly_probe_position "
        "WHERE probe_id=? AND entity_id=? AND asym_id=? AND residue_number=?",
        (probe_id, entity_id, chain, residue),
    ).fetchone()
    if row:
        return int(row["id"])
    with db.conn:
        cursor = db.conn.execute(
            "INSERT INTO flr_poly_probe_position "
            "(probe_id, entity_id, asym_id, residue_number, residue_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (probe_id, entity_id, chain, residue, "DA"),
        )
        return cursor.lastrowid


def _upsert_experiment_type(db: MFDatabase, name: str, category: str, description: str) -> int:
    row = db.conn.execute(
        "SELECT type_id FROM flr_experiment_type WHERE name=?", (name,)
    ).fetchone()
    if row:
        return int(row["type_id"])
    return db.add_experiment_type(name, category=category, description=description)


def _ensure_probe_types(db: MFDatabase) -> dict[str, int]:
    existing = {r["type_name"]: int(r["type_id"]) for r in db.get_probe_types()}
    for name, desc in [("organic_dye", "Organic dye"),
                        ("amino_acid", "Amino acid fluorophore"),
                        ("nucleic_acid", "Nucleic acid fluorophore")]:
        if name not in existing:
            with db.conn:
                db.conn.execute(
                    "INSERT OR IGNORE INTO probe_types (type_name, display_name) VALUES (?, ?)",
                    (name, desc),
                )
            row = db.conn.execute(
                "SELECT type_id FROM probe_types WHERE type_name=?", (name,)
            ).fetchone()
            existing[name] = int(row["type_id"]) if row else len(existing) + 1
    return existing


def _seed_flr_tables(db: MFDatabase) -> tuple[int, int, int, int]:
    """Seed the legacy flr_* tables with demo entities, returning key ids."""
    type_ids = _ensure_probe_types(db)
    dye_type = type_ids.get("organic_dye", 1)

    probe_a488 = _upsert_probe(
        db, "Alexa488", dye_type, category="organic_dye",
        abs_max=495.0, em_max=519.0, qy=0.92, ext_coeff=73000.0,
    )
    probe_a647 = _upsert_probe(
        db, "Alexa647", dye_type, category="organic_dye",
        abs_max=650.0, em_max=665.0, qy=0.33, ext_coeff=250000.0,
    )

    d = DEMO
    _upsert_user(db, d["user_id"], "John Doe", "john.doe@example.org")
    _upsert_device(db, d["device_id"], "MT200", device_type="TCSPC", model="MT200", serial="MT200-001")
    _upsert_entity(db, d["entity_id"], DNA_SEQ)
    _upsert_entity_assembly(db, d["assembly_id"],
                            "Demo DNA assembly labeled with Alexa488/Alexa647")
    _upsert_sample_condition(db, d["condition_id"], ph=7.4, temperature=298.15, buffer="PBS")

    pos_a488 = _upsert_poly_probe_position(db, probe_a488, d["entity_id"], 10)
    pos_a647 = _upsert_poly_probe_position(db, probe_a647, d["entity_id"], 20)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    if not db.get_sample(d["sample_id"]):
        db.add_sample(
            d["sample_id"],
            uuid=str(_uuid.uuid4()),
            description="Demo DNA labeled with Alexa488/Alexa647 for smFRET",
            details="Example smFRET sample using BH SPC-132 test data",
            num_of_probes=2,
            solvent_phase="liquid",
            sample_condition_id=d["condition_id"],
            entity_assembly_id=d["assembly_id"],
            project_id=d["project_id"],
            measured_by_user_id=d["user_id"],
            measured_by_device_id=d["device_id"],
            measured_at=now,
        )
        logger.info("Created sample %s", d["sample_id"])

    if not db.get_sample_probe_mappings(d["sample_id"]):
        db.add_sample_probe(d["sample_id"], probe_a488, "donor",
                            description="Donor (Alexa488) at position 10",
                            poly_probe_position_id=pos_a488)
        db.add_sample_probe(d["sample_id"], probe_a647, "acceptor",
                            description="Acceptor (Alexa647) at position 20",
                            poly_probe_position_id=pos_a647)
        logger.info("Created sample-probe mappings")

    exp_type_id = _upsert_experiment_type(
        db, "single_molecule_alex", "Single-molecule",
        "Alternating-laser excitation single-molecule experiment",
    )

    if not db.get_experiment(d["experiment_id"]):
        db.add_experiment(
            d["experiment_id"],
            type_id=exp_type_id,
            sample_id=d["sample_id"],
            project_id=d["project_id"],
            measured_by_user_id=d["user_id"],
            measured_by_device_id=d["device_id"],
            started_at=now,
            status="completed",
            details="smFRET measurement of demo DNA on MT200 with burst selection",
        )
        logger.info("Created experiment %s", d["experiment_id"])

    return probe_a488, probe_a647, pos_a488, pos_a647


def seed_example(db_path: str | Path | None = None) -> dict[str, object]:
    """Populate the MFDB with example smFRET demo data.

    Parameters
    ----------
    db_path : str or pathlib.Path, optional
        Database path.  Defaults to the user database resolved by
        :func:`resolve_database_path`.

    Returns
    -------
    dict
        Summary of seeded records and source test data.
    """
    path = Path(db_path) if db_path is not None else resolve_database_path()
    logging.basicConfig(level=logging.INFO)
    d = DEMO
    summary: dict[str, object] = {
        "database_path": str(path),
        "sample_id": d["sample_id"],
        "experiment_id": d["experiment_id"],
        "processing_id": d["processing_id"],
        "source_data_dir": str(SPC_DATA_DIR),
        "raw_data_ids": [],
        "quality_example_raw_data_ids": [],
        "processed_data_id": None,
        "used_test_files": [],
    }

    with MFDatabase(path) as db:
        _seed_flr_tables(db)

        if not SPC_DATA_DIR.is_dir():
            logger.warning("SPC test data not found at %s -- skipping MFDB seeding", SPC_DATA_DIR)
            summary["warning"] = f"SPC test data not found at {SPC_DATA_DIR}"
            return summary

        raw_ids: list[str] = []
        for i in range(3):
            spc_path = _spc_path(i)
            if not spc_path.exists():
                continue
            summary["used_test_files"].append(str(spc_path))
            raw_id = f"raw_demo_sm_dna_{i:03d}"
            if db.conn.execute(
                "SELECT 1 FROM mfdb_artifact WHERE artifact_id=?", (raw_id,)
            ).fetchone():
                raw_ids.append(raw_id)
                continue
            db.add_raw_data_reference(
                raw_data_id=raw_id,
                experiment_id=d["experiment_id"],
                data_type="SPC",
                storage_mode="local_file",
                file_path=str(spc_path.resolve()),
                size_bytes=spc_path.stat().st_size,
                checksum=f"demo:{spc_path.name}",
            )
            raw_ids.append(raw_id)
            logger.info("Registered raw data %s (%s, %d bytes)",
                        raw_id, spc_path.name, spc_path.stat().st_size)
            link_artifact_to_sample(db, raw_id, d["sample_id"])

        for raw_id in raw_ids:
            link_artifact_to_sample(db, raw_id, d["sample_id"])

        if not raw_ids:
            logger.warning("No SPC files found at %s -- skipping processing run", SPC_DATA_DIR)
            summary["warning"] = f"No SPC files found at {SPC_DATA_DIR}"
            return summary
        summary["raw_data_ids"] = raw_ids

        unlinked_path = _spc_path(0)
        if unlinked_path.exists():
            unlinked_raw_id = "raw_demo_unlinked_sample_red_flag"
            if not db.conn.execute(
                "SELECT 1 FROM mfdb_artifact WHERE artifact_id=?", (unlinked_raw_id,)
            ).fetchone():
                db.add_raw_data_reference(
                    raw_data_id=unlinked_raw_id,
                    experiment_id=d["experiment_id"],
                    data_type="SPC",
                    storage_mode="local_file",
                    file_path=str(unlinked_path.resolve()),
                    size_bytes=unlinked_path.stat().st_size,
                    checksum=f"demo:unlinked:{unlinked_path.name}",
                    validation_status="unvalidated",
                )
                logger.info("Registered unlinked red-flag raw data %s", unlinked_raw_id)
            summary["quality_example_raw_data_ids"] = [unlinked_raw_id]

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        # --- Processing run (direct SQL to bypass PK mismatch in record_operation_link) ---
        if not db.conn.execute(
            "SELECT 1 FROM mfdb_operation WHERE operation_id=?", (d["processing_id"],)
        ).fetchone():
            settings = {
                "photon_filter": {"routing_channels": [0, 1]},
                "burst_detection": {
                    "method": "sliding_window",
                    "window_width": 1000,
                    "threshold": 10,
                },
                "correction": {"gamma": 1.0, "beta": 1.0, "dir": 0.0},
            }
            oid = d["processing_id"]
            with db._transaction():
                db.record_operation(
                    operation_id=oid,
                    operation_type="burst_selection",
                    experiment_id=d["experiment_id"],
                    operator_user_id=d["user_id"],
                    settings=settings,
                    software_module="chisurf.plugins.burst.burst_selection",
                    software_version="1.0",
                    status="succeeded",
                    started_at=now,
                    ended_at=now,
                )
                # Link input raw data artifacts (use INSERT OR REPLACE to
                # match the actual PK = (operation_id, artifact_id, direction))
                for raw_id in raw_ids:
                    db.conn.execute(
                        "INSERT OR REPLACE INTO mfdb_operation_artifact "
                        "(operation_id, artifact_id, direction, role) "
                        "VALUES (?, ?, ?, ?)",
                        (oid, raw_id, "input", "raw_data"),
                    )
                db.add_audit_log(
                    action="create",
                    target_type="processing_run",
                    target_id=oid,
                    operator_user_id=d["user_id"],
                    details={"experiment_id": d["experiment_id"], "processing_type": "burst_selection"},
                )
            logger.info("Created processing run %s", oid)

        # --- Processed data product (direct SQL) ---
        prod_id = f"prod_{d['processing_id']}"
        if not db.conn.execute(
            "SELECT 1 FROM mfdb_artifact WHERE artifact_id=?", (prod_id,)
        ).fetchone():
            demo_dir = Path.home() / ".chisurf" / "flr" / "demo"
            demo_dir.mkdir(parents=True, exist_ok=True)
            bur_path = demo_dir / f"{d['processing_id']}.bur"
            with db._transaction():
                db.register_artifact(
                    artifact_id=prod_id,
                    artifact_kind="bur",
                    storage_mode="local_file",
                    file_path=str(bur_path.resolve()),
                    row_count=5577,
                    validation_status="valid",
                    checksum=f"demo:{bur_path.name}",
                )
                db.conn.execute(
                    "INSERT OR REPLACE INTO mfdb_operation_artifact "
                    "(operation_id, artifact_id, direction, role) "
                    "VALUES (?, ?, ?, ?)",
                    (d["processing_id"], prod_id, "output", "bur"),
                )
                db.add_audit_log(
                    action="create",
                    target_type="processed_data",
                    target_id=prod_id,
                    details={"processing_id": d["processing_id"], "product_type": "bur"},
                )
            logger.info("Created processed data %s", prod_id)
        link_artifact_to_sample(db, prod_id, d["sample_id"])
        summary["processed_data_id"] = prod_id

    logger.info("Seeding complete at %s", path)
    return summary


if __name__ == "__main__":
    seed_example()
