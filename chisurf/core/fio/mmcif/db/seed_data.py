"""Curated seed data for the ChiSurf fluorescence sample database."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .repository import FluorescenceDatabase
from .database_resolver import source_database_path

T4_LYSOZYME_SEQUENCE = "MSTLQEK"


def seed_curated_database(db_path: Optional[str | Path] = None) -> Path:
    """Populate a sample database with curated fluorescence examples.

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
    db = FluorescenceDatabase(path)
    try:
        _seed_probe_types(db)
        _seed_probes(db)
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


def _seed_probe_types(db: FluorescenceDatabase) -> None:
    db.add_probe_type("organic_dye", "Organic dye")
    db.add_probe_type("amino_acid", "Amino acid fluorophore")
    db.add_probe_type("nucleic_acid", "Nucleic acid fluorophore")


def _seed_probes(db: FluorescenceDatabase) -> None:
    types = {row["type_name"]: row["type_id"] for row in db.get_probe_types()}
    probes = [
        ("Alexa488", "organic_dye", "organic_dye", 495.0, 519.0, 0.92, 73000.0),
        ("Alexa594", "organic_dye", "organic_dye", 590.0, 617.0, 0.66, 87000.0),
        ("Cy3", "organic_dye", "organic_dye", 550.0, 570.0, 0.15, 150000.0),
        ("Cy5", "organic_dye", "organic_dye", 649.0, 670.0, 0.27, 250000.0),
        ("ATTO647N", "organic_dye", "organic_dye", 644.0, 669.0, 0.65, 150000.0),
        ("Trp", "amino_acid", "protein", 280.0, 350.0, 0.13, None),
        ("2-aminopurine", "nucleic_acid", "other", 310.0, 370.0, 0.68, 23000.0),
    ]
    for name, type_name, category, abs_max, em_max, qy, ext_coeff in probes:
        probe_id = db.add_probe(name, types[type_name], category=category, is_curated=1)
        db.add_optical_property(probe_id, "abs_max", abs_max, unit="nm")
        db.add_optical_property(probe_id, "em_max", em_max, unit="nm")
        db.add_optical_property(probe_id, "qy", qy, unit="")
        if ext_coeff is not None:
            db.add_optical_property(probe_id, "ext_coeff", ext_coeff, unit="M-1 cm-1")
        x = np.array([abs_max - 40.0, abs_max, abs_max + 40.0], dtype=np.float64)
        y = np.array([0.25, 1.0, 0.25], dtype=np.float64)
        db.add_spectrum(probe_id, "absorption", x, y, details="Curated seed spectrum")
        y_em = np.array([0.1, 1.0, 0.2], dtype=np.float64)
        db.add_spectrum(
            probe_id, "emission", np.array([em_max - 35.0, em_max, em_max + 35.0]), y_em
        )


def _seed_entities(db: FluorescenceDatabase) -> None:
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


def _seed_conditions_and_assemblies(db: FluorescenceDatabase) -> None:
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


def _seed_users_and_devices(db: FluorescenceDatabase) -> None:
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


def _seed_positions(db: FluorescenceDatabase) -> None:
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


def _seed_samples(db: FluorescenceDatabase) -> None:
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


def _seed_experiment_types(db: FluorescenceDatabase) -> None:
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


def _seed_experiments(db: FluorescenceDatabase) -> None:
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


def _seed_analyses(db: FluorescenceDatabase) -> None:
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


def _probe_id(db: FluorescenceDatabase, name: str) -> int:
    row = db.conn.execute(
        "SELECT probe_id FROM probes WHERE chromophore_name=?", (name,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing probe {name}")
    return int(row["probe_id"])


def _position_id(
    db: FluorescenceDatabase, probe_name: str, entity_id: str, residue_number: int
) -> int:
    row = db.conn.execute(
        "SELECT id FROM flr_poly_probe_position WHERE probe_id=? AND entity_id=? AND residue_number=?",
        (_probe_id(db, probe_name), entity_id, residue_number),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing position {probe_name} {entity_id} {residue_number}")
    return int(row["id"])
