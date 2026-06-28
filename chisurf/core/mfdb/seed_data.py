"""Curated seed data for the ChiSurf fluorescence sample database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from chisurf.core.mfdb.repository import MFDatabase, _utc_now
from chisurf.core.mfdb.database_resolver import source_database_path

T4_LYSOZYME_SEQUENCE = "MSTLQEK"

_DATA_DIR = Path(__file__).resolve().parent / "data"

# Common FRET pairs for R0 precomputation
# (donor_name, acceptor_name) — names must match imported/chromophore_name
_COMMON_FRET_PAIRS = [
    ("ATTO 488", "ATTO 647N"),
    ("ATTO 532", "ATTO 647N"),
    ("ATTO 550", "ATTO 647N"),
    ("ATTO 565", "ATTO 647N"),
    ("Alexa488", "Alexa594"),
    ("Alexa488", "Alexa647"),
    ("Alexa555", "Alexa647"),
    ("Cy3", "Cy5"),
    ("Cy3B", "ATTO 647N"),
    ("Cy3B", "Cy5"),
]

# Additional common dyes to seed if not present in the reference import
_ADDITIONAL_PROBES = [
    {
        "name": "Trp", "type_name": "amino_acid", "category": "protein",
        "absorption_max_nm": 280.0, "emission_max_nm": 350.0,
        "quantum_yield": 0.13, "extinction_coefficient": 6990.0,
        "fwhm_abs": 30.0, "fwhm_em": 55.0,
    },
    {
        "name": "2-aminopurine", "type_name": "nucleic_acid", "category": "other",
        "absorption_max_nm": 310.0, "emission_max_nm": 370.0,
        "quantum_yield": 0.68, "extinction_coefficient": 23000.0,
        "fwhm_abs": 25.0, "fwhm_em": 40.0,
    },
]


def _gaussian_spectrum(center_nm, fwhm_nm, wl_start=250.0, wl_end=800.0):
    """Generate a Gaussian approximation of a spectrum (fallback)."""
    wl = np.arange(wl_start, wl_end + 1, 1.0)
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    intensity = np.exp(-0.5 * ((wl - center_nm) / sigma) ** 2)
    return wl, intensity


def seed_curated_database(db_path: Optional[str | Path] = None) -> Path:
    """Populate a sample database with curated fluorescence examples.

    Imports the scraped fluorophore reference set (spectra.db) for real
    spectral data, then seeds curated sample/entity/experiment records.

    Parameters
    ----------
    db_path : str or pathlib.Path, optional
        Database path. Defaults to the curated source database in ``src``.

    Returns
    -------
    pathlib.Path
        Database path.
    """
    path = Path(db_path) if db_path is not None else source_database_path()
    db = MFDatabase(path)
    try:
        # Import scraped reference data first (idempotent, dedup by name)
        counts = db.import_reference_set()
        if counts["probes"] > 0:
            import logging
            logging.getLogger(__name__).info(
                "Seeded %d probes, %d spectra, %d optical properties from reference set",
                counts["probes"], counts["spectra"], counts["optical_properties"],
            )
        _seed_probe_types(db)
        _seed_probes(db)
        _seed_forster_radii(db)
        _seed_entities(db)
        _seed_conditions_and_assemblies(db)
        _seed_users_and_devices(db)
        _seed_positions(db)
        _seed_samples(db)
        _seed_analyses(db)
        _seed_experiment_types(db)
        _seed_experiments(db)
    finally:
        db.close()
    return path


def _seed_probe_types(db: MFDatabase) -> None:
    db.add_probe_type("organic_dye", "Organic dye")
    db.add_probe_type("amino_acid", "Amino acid fluorophore")
    db.add_probe_type("nucleic_acid", "Nucleic acid fluorophore")


def _load_probe_properties() -> list[dict]:
    """Load probe properties from JSON file, with fallback to hardcoded values."""
    filepath = _DATA_DIR / "probe_properties.json"
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return [
        {
            "name": "Alexa488", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 495.0, "emission_max_nm": 519.0,
            "quantum_yield": 0.92, "extinction_coefficient": 73000.0,
            "fwhm_abs": 28.0, "fwhm_em": 37.0,
        },
        {
            "name": "Alexa546", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 556.0, "emission_max_nm": 573.0,
            "quantum_yield": 0.79, "extinction_coefficient": 104000.0,
            "fwhm_abs": 25.0, "fwhm_em": 33.0,
        },
        {
            "name": "Alexa555", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 555.0, "emission_max_nm": 565.0,
            "quantum_yield": 0.10, "extinction_coefficient": 155000.0,
            "fwhm_abs": 27.0, "fwhm_em": 38.0,
        },
        {
            "name": "Alexa568", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 578.0, "emission_max_nm": 603.0,
            "quantum_yield": 0.69, "extinction_coefficient": 91300.0,
            "fwhm_abs": 28.0, "fwhm_em": 40.0,
        },
        {
            "name": "Alexa594", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 590.0, "emission_max_nm": 617.0,
            "quantum_yield": 0.66, "extinction_coefficient": 87000.0,
            "fwhm_abs": 28.0, "fwhm_em": 40.0,
        },
        {
            "name": "Alexa647", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 650.0, "emission_max_nm": 665.0,
            "quantum_yield": 0.33, "extinction_coefficient": 270000.0,
            "fwhm_abs": 28.0, "fwhm_em": 35.0,
        },
        {
            "name": "Cy3", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 550.0, "emission_max_nm": 570.0,
            "quantum_yield": 0.15, "extinction_coefficient": 150000.0,
            "fwhm_abs": 25.0, "fwhm_em": 35.0,
        },
        {
            "name": "Cy3B", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 558.0, "emission_max_nm": 572.0,
            "quantum_yield": 0.67, "extinction_coefficient": 130000.0,
            "fwhm_abs": 25.0, "fwhm_em": 33.0,
        },
        {
            "name": "Cy5", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 649.0, "emission_max_nm": 670.0,
            "quantum_yield": 0.27, "extinction_coefficient": 250000.0,
            "fwhm_abs": 25.0, "fwhm_em": 35.0,
        },
        {
            "name": "Cy5.5", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 673.0, "emission_max_nm": 707.0,
            "quantum_yield": 0.23, "extinction_coefficient": 209000.0,
            "fwhm_abs": 25.0, "fwhm_em": 40.0,
        },
        {
            "name": "ATTO 488", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 501.0, "emission_max_nm": 523.0,
            "quantum_yield": 0.80, "extinction_coefficient": 90000.0,
            "fwhm_abs": 22.0, "fwhm_em": 30.0,
        },
        {
            "name": "ATTO 532", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 532.0, "emission_max_nm": 553.0,
            "quantum_yield": 0.90, "extinction_coefficient": 115000.0,
            "fwhm_abs": 23.0, "fwhm_em": 32.0,
        },
        {
            "name": "ATTO 550", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 554.0, "emission_max_nm": 576.0,
            "quantum_yield": 0.80, "extinction_coefficient": 120000.0,
            "fwhm_abs": 23.0, "fwhm_em": 32.0,
        },
        {
            "name": "ATTO 565", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 563.0, "emission_max_nm": 592.0,
            "quantum_yield": 0.90, "extinction_coefficient": 120000.0,
            "fwhm_abs": 23.0, "fwhm_em": 35.0,
        },
        {
            "name": "ATTO 590", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 594.0, "emission_max_nm": 624.0,
            "quantum_yield": 0.80, "extinction_coefficient": 120000.0,
            "fwhm_abs": 24.0, "fwhm_em": 36.0,
        },
        {
            "name": "ATTO 594", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 601.0, "emission_max_nm": 627.0,
            "quantum_yield": 0.85, "extinction_coefficient": 120000.0,
            "fwhm_abs": 25.0, "fwhm_em": 35.0,
        },
        {
            "name": "ATTO 647N", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 644.0, "emission_max_nm": 669.0,
            "quantum_yield": 0.65, "extinction_coefficient": 150000.0,
            "fwhm_abs": 22.0, "fwhm_em": 30.0,
        },
        {
            "name": "ATTO 655", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 663.0, "emission_max_nm": 684.0,
            "quantum_yield": 0.30, "extinction_coefficient": 125000.0,
            "fwhm_abs": 22.0, "fwhm_em": 28.0,
        },
        {
            "name": "ATTO 680", "type_name": "organic_dye", "category": "organic_dye",
            "absorption_max_nm": 680.0, "emission_max_nm": 700.0,
            "quantum_yield": 0.30, "extinction_coefficient": 125000.0,
            "fwhm_abs": 23.0, "fwhm_em": 30.0,
        },
        {
            "name": "Trp", "type_name": "amino_acid", "category": "protein",
            "absorption_max_nm": 280.0, "emission_max_nm": 350.0,
            "quantum_yield": 0.13, "extinction_coefficient": 6990.0,
            "fwhm_abs": 30.0, "fwhm_em": 55.0,
        },
        {
            "name": "2-aminopurine", "type_name": "nucleic_acid", "category": "other",
            "absorption_max_nm": 310.0, "emission_max_nm": 370.0,
            "quantum_yield": 0.68, "extinction_coefficient": 23000.0,
            "fwhm_abs": 25.0, "fwhm_em": 40.0,
        },
    ]


def _seed_probes(db: MFDatabase) -> None:
    types = {row["type_name"]: row["type_id"] for row in db.get_probe_types()}
    probes_data = _load_probe_properties()

    for probe_data in probes_data:
        name = probe_data["name"]
        type_name = probe_data["type_name"]
        category = probe_data["category"]
        abs_max = probe_data["absorption_max_nm"]
        em_max = probe_data["emission_max_nm"]
        qy = probe_data["quantum_yield"]
        ext_coeff = probe_data["extinction_coefficient"]
        fwhm_abs = probe_data.get("fwhm_abs", 28.0)
        fwhm_em = probe_data.get("fwhm_em", 35.0)

        # Check if already imported from reference set
        existing = db.conn.execute(
            "SELECT probe_id FROM probes WHERE chromophore_name = ? AND deleted_at IS NULL",
            (name,),
        ).fetchone()

        if existing:
            probe_id = int(existing["probe_id"])
            # Mark as approved if it was imported as unverified
            db.approve_probe(probe_id, verified_by="seed_data")
            # Ensure optical properties are present
            props = {r["property_name"] for r in db.conn.execute(
                "SELECT property_name FROM optical_properties WHERE probe_id = ? AND deleted_at IS NULL",
                (probe_id,),
            ).fetchall()}
            if "abs_max" not in props:
                db.add_optical_property(probe_id, "abs_max", abs_max, unit="nm")
            if "em_max" not in props:
                db.add_optical_property(probe_id, "em_max", em_max, unit="nm")
            if "qy" not in props:
                db.add_optical_property(probe_id, "qy", qy, unit="")
            if ext_coeff is not None and "ext_coeff" not in props:
                db.add_optical_property(probe_id, "ext_coeff", ext_coeff, unit="M-1 cm-1")
        else:
            probe_id = db.add_probe(name, types[type_name], category=category, is_curated=1)
            with db.conn:
                db.conn.execute(
                    "UPDATE probes SET verification_status = 'approved', quality = 'high' WHERE probe_id = ?",
                    (probe_id,),
                )
            db.add_optical_property(probe_id, "abs_max", abs_max, unit="nm")
            db.add_optical_property(probe_id, "em_max", em_max, unit="nm")
            db.add_optical_property(probe_id, "qy", qy, unit="")
            if ext_coeff is not None:
                db.add_optical_property(probe_id, "ext_coeff", ext_coeff, unit="M-1 cm-1")

            # Add fallback Gaussian spectra if no spectra exist from import
            has_abs = db.conn.execute(
                "SELECT 1 FROM spectra WHERE probe_id = ? AND spectrum_type = 'absorption' AND deleted_at IS NULL",
                (probe_id,),
            ).fetchone()
            has_em = db.conn.execute(
                "SELECT 1 FROM spectra WHERE probe_id = ? AND spectrum_type = 'emission' AND deleted_at IS NULL",
                (probe_id,),
            ).fetchone()
            if not has_abs:
                x, y = _gaussian_spectrum(abs_max, fwhm_abs)
                db.add_spectrum(probe_id, "absorption", x, y, details="Fallback Gaussian spectrum")
            if not has_em:
                x, y = _gaussian_spectrum(em_max, fwhm_em)
                db.add_spectrum(probe_id, "emission", x, y, details="Fallback Gaussian spectrum")


def _seed_forster_radii(db: MFDatabase) -> None:
    """Precompute Forster radii for common FRET pairs (PRD-06 Task 4)."""
    from chisurf.core.fluorescence.fret.forster import (
        compute_forster_radius_from_spectra,
    )

    for donor_name, acceptor_name in _COMMON_FRET_PAIRS:
        donor_spec = db.get_spectra_for_forster(donor_name)
        acceptor_spec = db.get_spectra_for_forster(acceptor_name)
        if not donor_spec or not acceptor_spec:
            continue
        if "emission" not in donor_spec or "absorption" not in acceptor_spec:
            continue

        donor_row = db.conn.execute(
            "SELECT probe_id FROM probes WHERE chromophore_name = ? AND deleted_at IS NULL",
            (donor_name,),
        ).fetchone()
        acceptor_row = db.conn.execute(
            "SELECT probe_id FROM probes WHERE chromophore_name = ? AND deleted_at IS NULL",
            (acceptor_name,),
        ).fetchone()
        if not donor_row or not acceptor_row:
            continue

        donor_props = {r["property_name"]: r["property_value"] for r in db.conn.execute(
            "SELECT property_name, property_value FROM optical_properties WHERE probe_id = ? AND deleted_at IS NULL",
            (int(donor_row["probe_id"]),),
        ).fetchall()}
        qy_str = donor_props.get("qy", "0.0")
        try:
            donor_qy = float(str(qy_str).replace(",", ""))
        except (ValueError, TypeError):
            continue

        em_wl = donor_spec["emission"]["wavelengths"]
        em_int = donor_spec["emission"]["intensity"]
        abs_wl = acceptor_spec["absorption"]["wavelengths"]
        abs_int = acceptor_spec["absorption"]["intensity"]

        # Scale absorption intensity by extinction coefficient
        acc_props = {r["property_name"]: r["property_value"] for r in db.conn.execute(
            "SELECT property_name, property_value FROM optical_properties WHERE probe_id = ? AND deleted_at IS NULL",
            (int(acceptor_row["probe_id"]),),
        ).fetchall()}
        ec_str = acc_props.get("ext_coeff", "0")
        try:
            ext_coeff = float(str(ec_str).replace(",", ""))
        except (ValueError, TypeError):
            ext_coeff = 100000.0  # fallback

        eps_a = abs_int * ext_coeff

        # Both spectra must be on a common wavelength grid
        # Interpolate onto 1 nm grid covering the overlap region
        wl_min = max(float(em_wl[0]), float(abs_wl[0]))
        wl_max = min(float(em_wl[-1]), float(abs_wl[-1]))
        if wl_min >= wl_max:
            continue
        common_wl = np.arange(wl_min, wl_max + 1, 1.0)

        from chisurf.core.fluorescence.fret.forster import (
            overlap_integral,
            forster_radius,
        )

        try:
            fd = np.interp(common_wl, em_wl, em_int)
            ea = np.interp(common_wl, abs_wl, abs_int * ext_coeff)
            J = overlap_integral(common_wl, fd, ea)
            R0 = forster_radius(J, donor_quantum_yield=donor_qy)
        except Exception:
            continue

        if R0 <= 0:
            continue

        # Generate a deterministic forster_radius_id
        fr_id = f"seed_{donor_name}_{acceptor_name}".replace(" ", "_").replace(".", "_")

        # Use the first sample that has both probes, or NULL
        sample_id = None
        try:
            sample_row = db.conn.execute(
                """SELECT sp1.sample_id
                   FROM flr_sample_probe sp1
                   JOIN flr_sample_probe sp2 ON sp1.sample_id = sp2.sample_id
                   WHERE sp1.probe_id = ? AND sp2.probe_id = ?
                   LIMIT 1""",
                (int(donor_row["probe_id"]), int(acceptor_row["probe_id"])),
            ).fetchone()
            if sample_row:
                sample_id = sample_row[0]
        except Exception:
            pass

        with db.conn:
            db.conn.execute(
                """INSERT OR REPLACE INTO flr_fret_forster_radius
                   (forster_radius_id, sample_id, donor_probe_id, acceptor_probe_id,
                    forster_radius, kappa_squared, index_of_refraction, overlap_integral,
                    details, created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fr_id,
                    sample_id,
                    int(donor_row["probe_id"]),
                    int(acceptor_row["probe_id"]),
                    float(R0),
                    2.0 / 3.0,
                    1.33,
                    float(J),
                    f"Precomputed from seed data (PRD-06 Task 4); donor QY={donor_qy}",
                    _utc_now(),
                    _utc_now(),
                    None,
                ),
            )


def _seed_entities(db: MFDatabase) -> None:
    db.add_entity(
        "148L",
        type="polymer",
        description="T4 lysozyme curated FRET sample",
        common_name="T4 lysozyme",
    )
    db.set_sequence("148L", list("MKWVTFISLLFLFSSAYS"))
    db.add_entity(
        "1RTD_DNA",
        type="polymer",
        description="HIV reverse transcriptase DNA duplex seed",
        common_name="HIV RT DNA",
    )
    db.set_sequence("1RTD_DNA", list("ACGTACGTACGTACGTACGT"))


def _seed_conditions_and_assemblies(db: MFDatabase) -> None:
    db.add_sample_condition(
        "pbs_ph74_25c",
        ph=7.4,
        temperature=298.15,
        ionic_strength=0.15,
        buffer_composition="PBS",
        details="PBS pH 7.4 at 25 C",
    )
    db.add_sample_condition(
        "hepes_50mm_nacl_ph70_20c",
        ph=7.0,
        temperature=293.15,
        ionic_strength=0.05,
        buffer_composition="20 mM HEPES, 50 mM NaCl",
        details="Low-salt FRET buffer at 20 C",
    )
    db.add_entity_assembly(
        "148L_65_141_double_cys",
        description="T4 lysozyme positions 65 and 141 labeled with donor and acceptor",
    )
    db.add_entity_assembly(
        "1RTD_DNA_10_20_duplex",
        description="DNA duplex labeled at positions 10 and 20",
    )


def _seed_users_and_devices(db: MFDatabase) -> None:
    db.add_user("thomas", "Thomas Peulen", "thomas.peulen@tu-dortmund.de", "TU Dortmund")
    db.add_user("operator_demo", "Demo Operator", "operator@example.org", "Demo lab")
    db.add_device(
        "spectrometer_demo",
        "Demo TCSPC spectrometer",
        device_type="TCSPC",
        model="DemoSpec 3000",
        serial_number="DS3000-001",
        location="TU Dortmund / Fluorescence lab",
        owner="FRET group",
        details="Curated example instrument",
    )
    db.add_device(
        "plate_reader_demo",
        "Demo plate reader",
        device_type="plate_reader",
        model="ReaderX",
        serial_number="RX-001",
        location="TU Dortmund / Screening lab",
        owner="FRET group",
        details="Curated example device",
    )


def _seed_positions(db: MFDatabase) -> None:
    alexa488 = _probe_id(db, "Alexa488")
    alexa594 = _probe_id(db, "Alexa594")
    cy3 = _probe_id(db, "Cy3")
    cy5 = _probe_id(db, "Cy5")
    trp = _probe_id(db, "Trp")
    ap = _probe_id(db, "2-aminopurine")
    db.add_poly_probe_position(
        alexa488, "148L", 65, asym_id="A", residue_name="CYS", description="Donor label site"
    )
    db.add_poly_probe_position(
        alexa594, "148L", 141, asym_id="A", residue_name="CYS", description="Acceptor label site"
    )
    db.add_poly_probe_position(
        trp, "148L", 126, asym_id="A", residue_name="TRP", description="Intrinsic Trp reference"
    )
    db.add_poly_probe_position(
        cy3, "1RTD_DNA", 10, asym_id="P", residue_name="DA", description="DNA donor label site"
    )
    db.add_poly_probe_position(
        cy5, "1RTD_DNA", 20, asym_id="P", residue_name="DC", description="DNA acceptor label site"
    )
    db.add_poly_probe_position(
        ap, "1RTD_DNA", 12, asym_id="P", residue_name="DA", description="2-aminopurine reference"
    )


def _seed_samples(db: MFDatabase) -> None:
    db.add_sample(
        "148L_65_141_A488_A594",
        uuid="00000000-0000-4000-8000-000000000001",
        description="T4 lysozyme 65/141 Alexa488/Alexa594 FRET sample",
        details="Curated protein FRET sample for plugin testing.",
        num_of_probes=2,
        solvent_phase="liquid",
        sample_condition_id="pbs_ph74_25c",
        entity_assembly_id="148L_65_141_double_cys",
        project_id="demo_fret",
        measured_by_user_id="thomas",
        measured_by_device_id="spectrometer_demo",
        measured_at="2026-06-11T09:00:00",
    )
    db.set_sample_key_value(
        "148L_65_141_A488_A594",
        "pdbx.sample_type",
        "protein",
        "PDBx sample metadata imported from curated seed data",
    )
    db.set_sample_key_value(
        "148L_65_141_A488_A594",
        "lims.batch_id",
        "BATCH-148L-001",
        "Example LIMS batch identifier",
    )
    db.add_sample(
        "1RTD_DNA_10_20_Cy3_Cy5",
        uuid="00000000-0000-4000-8000-000000000002",
        description="HIV RT DNA duplex Cy3/Cy5 FRET sample",
        details="Curated DNA FRET sample for plugin testing.",
        num_of_probes=2,
        solvent_phase="liquid",
        sample_condition_id="hepes_50mm_nacl_ph70_20c",
        entity_assembly_id="1RTD_DNA_10_20_duplex",
        project_id="demo_fret",
        measured_by_user_id="operator_demo",
        measured_by_device_id="plate_reader_demo",
        measured_at="2026-06-11T10:00:00",
    )
    db.set_sample_key_value(
        "1RTD_DNA_10_20_Cy3_Cy5",
        "pdbihm.entry_id",
        "1RTD",
        "PDB-IHM/PDBx sample metadata imported from curated seed data",
    )
    db.set_sample_key_value(
        "1RTD_DNA_10_20_Cy3_Cy5",
        "lims.batch_id",
        "BATCH-1RTD-001",
        "Example LIMS batch identifier",
    )
    db.add_sample(
        "148L_intrinsic_trp",
        uuid="00000000-0000-4000-8000-000000000003",
        description="T4 lysozyme intrinsic tryptophan sample",
        details="Curated intrinsic protein fluorescence sample.",
        num_of_probes=1,
        solvent_phase="liquid",
        sample_condition_id="pbs_ph74_25c",
        entity_assembly_id="148L_65_141_double_cys",
        project_id="demo_fret",
        measured_by_user_id="thomas",
        measured_by_device_id="spectrometer_demo",
        measured_at="2026-06-11T11:00:00",
    )
    db.set_sample_key_value(
        "148L_intrinsic_trp",
        "flrcif.sample_class",
        "intrinsic",
        "FLR CIF sample metadata imported from curated seed data",
    )


def _seed_experiment_types(db: MFDatabase) -> None:
    experiment_types = [
        ("imaging", "Imaging", "Optical fluorescence imaging experiment"),
        ("flim", "Imaging", "Fluorescence lifetime imaging microscopy"),
        ("single_molecule_alex", "Single-molecule", "Alternating-laser excitation single-molecule experiment"),
        ("single_molecule_mfd", "Single-molecule", "Multiparameter fluorescence detection experiment"),
        ("tcspc", "Time-resolved", "Time-correlated single photon counting decay experiment"),
        ("fcs", "Fluctuation", "Fluorescence correlation spectroscopy experiment"),
        ("spectra", "Spectroscopy", "Steady-state or time-resolved spectral experiment"),
    ]
    for name, category, description in experiment_types:
        db.add_experiment_type(name, category=category, description=description)


def _seed_experiments(db: MFDatabase) -> None:
    type_ids = {row["name"]: row["type_id"] for row in db.get_experiment_types()}
    db.add_experiment(
        "148L_tcspc_001",
        type_id=type_ids["tcspc"],
        sample_id="148L_65_141_A488_A594",
        project_id="demo_fret",
        measured_by_user_id="thomas",
        measured_by_device_id="spectrometer_demo",
        started_at="2026-06-11T09:15:00",
        status="completed",
        details="Curated TCSPC decay experiment linked to the 148L FRET sample",
    )
    db.set_experiment_key_value(
        "148L_tcspc_001",
        "flrcif.experiment_type",
        "tcspc",
        "FLR CIF-compatible experiment type",
    )
    db.add_experiment_data(
        "148L_tcspc_001",
        data_type="tcspc",
        storage_mode="link",
        file_path="raw/148L/148L_tcspc_001.ptu",
        mime_type="application/octet-stream",
        checksum="demo:148L_tcspc_001.ptu",
        details="Raw TCSPC file path for later file-loader integration",
    )

    db.add_experiment(
        "148L_flim_001",
        type_id=type_ids["flim"],
        sample_id="148L_65_141_A488_A594",
        project_id="demo_fret",
        measured_by_user_id="thomas",
        measured_by_device_id="spectrometer_demo",
        started_at="2026-06-11T09:45:00",
        status="completed",
        details="Curated FLIM experiment linked to the 148L FRET sample",
    )
    db.add_experiment_data(
        "148L_flim_001",
        data_type="flim",
        storage_mode="folder",
        folder_path="raw/148L/flim_001/",
        details="FLIM raw data folder for later file-loader integration",
    )

    spectra_experiment_id = "1RTD_spectra_001"
    db.add_experiment(
        spectra_experiment_id,
        type_id=type_ids["spectra"],
        sample_id="1RTD_DNA_10_20_Cy3_Cy5",
        project_id="demo_fret",
        measured_by_user_id="operator_demo",
        measured_by_device_id="plate_reader_demo",
        started_at="2026-06-11T10:15:00",
        status="completed",
        details="Curated emission spectra experiment linked to the 1RTD DNA sample",
    )
    db.set_experiment_key_value(
        spectra_experiment_id,
        "flrcif.data_type",
        "spectra",
        "FLR CIF-compatible data type",
    )
    db.add_experiment_data(
        spectra_experiment_id,
        data_type="spectra",
        storage_mode="embedded",
        mime_type="application/json",
        data_json='{"wavelengths_nm":[580,600,620,640,660,680],"intensities":[0.1,0.35,0.9,1.0,0.72,0.25]}',
        reading_options_json='{"skiprows":0,"delimiter":","}',
        details="Small embedded spectra payload for LIMS-style preservation",
    )

    db.add_experiment(
        "148L_alex_001",
        type_id=type_ids["single_molecule_alex"],
        sample_id="148L_65_141_A488_A594",
        project_id="demo_fret",
        measured_by_user_id="thomas",
        measured_by_device_id="spectrometer_demo",
        started_at="2026-06-11T11:15:00",
        status="planned",
        details="Curated single-molecule ALEX experiment placeholder",
    )


def _seed_analyses(db: MFDatabase) -> None:
    a488_pos = _position_id(db, "Alexa488", "148L", 65)
    a594_pos = _position_id(db, "Alexa594", "148L", 141)
    sp1 = db.add_sample_probe(
        "148L_65_141_A488_A594", _probe_id(db, "Alexa488"), a488_pos, "donor", "Donor at 65"
    )
    sp2 = db.add_sample_probe(
        "148L_65_141_A488_A594", _probe_id(db, "Alexa594"), a594_pos, "acceptor", "Acceptor at 141"
    )
    db.add_sample_probe(
        "148L_intrinsic_trp",
        _probe_id(db, "Trp"),
        _position_id(db, "Trp", "148L", 126),
        "donor",
        "Intrinsic Trp",
    )
    cy3_pos = _position_id(db, "Cy3", "1RTD_DNA", 10)
    cy5_pos = _position_id(db, "Cy5", "1RTD_DNA", 20)
    sp3 = db.add_sample_probe(
        "1RTD_DNA_10_20_Cy3_Cy5", _probe_id(db, "Cy3"), cy3_pos, "donor", "DNA donor"
    )
    sp4 = db.add_sample_probe(
        "1RTD_DNA_10_20_Cy3_Cy5", _probe_id(db, "Cy5"), cy5_pos, "acceptor", "DNA acceptor"
    )
    db.add_sample_probe(
        "1RTD_DNA_10_20_Cy3_Cy5",
        _probe_id(db, "2-aminopurine"),
        _position_id(db, "2-aminopurine", "1RTD_DNA", 12),
        "donor",
        "2AP reference",
    )

    db.update_analysis_record(
        "148L_A488_A594_fret",
        sample_id="148L_65_141_A488_A594",
        type="intensity-based",
        method="steady-state FRET",
        sample_probe_id_1=sp1,
        sample_probe_id_2=sp2,
        details="Curated seed analysis",
    )
    db.update_analysis_record(
        "1RTD_DNA_Cy3_Cy5_fret",
        sample_id="1RTD_DNA_10_20_Cy3_Cy5",
        type="intensity-based",
        method="DNA FRET",
        sample_probe_id_1=sp3,
        sample_probe_id_2=sp4,
        details="Curated seed analysis",
    )


def _probe_id(db: MFDatabase, name: str) -> int:
    row = db.conn.execute(
        "SELECT probe_id FROM probes WHERE chromophore_name=?", (name,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing probe {name}")
    return int(row["probe_id"])


def _position_id(
    db: MFDatabase, probe_name: str, entity_id: str, residue_number: int
) -> int:
    row = db.conn.execute(
        "SELECT id FROM flr_poly_probe_position WHERE probe_id=? AND entity_id=? AND residue_number=?",
        (_probe_id(db, probe_name), entity_id, residue_number),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing position {probe_name} {entity_id} {residue_number}")
    return int(row["id"])
