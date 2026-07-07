"""Import PDBx, PDB-IHM, and FLR CIF data into the sample database."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from mfdb.repository import MFDatabase
logger = logging.getLogger(__name__)


def import_structure_file(db: MFDatabase, path: str | Path) -> Dict[str, Any]:
    """Import a PDBx/mmCIF, PDB-IHM CIF, or FLR CIF file into ``db``.

    Parameters
    ----------
    db : MFDatabase
        Target database.
    path : str or pathlib.Path
        Input CIF/mmCIF file.

    Returns
    -------
    dict
        Import summary with sample ids, entity ids, and warnings.
    """
    path = Path(path)
    summary: Dict[str, Any] = {"path": str(path), "samples": [], "entities": [], "warnings": []}
    systems = _read_ihm_systems(path, summary)
    if systems:
        _import_ihm_systems(db, systems, summary)
    _import_chisurf_extensions(db, path, summary)
    if not summary["samples"]:
        sample_id = _sample_id_from_path(path)
        db.add_sample(sample_id, description=path.stem, details=f"Imported from {path}")
        summary["samples"].append(sample_id)
    return summary


def _read_ihm_systems(path: Path, summary: Dict[str, Any]) -> List[Any]:
    """Read PDBx/IHM/FLR CIF using the ``ihm`` package."""
    try:
        import ihm.reader
    except Exception as exc:
        summary["warnings"].append(f"ihm reader unavailable: {exc}")
        return []

    try:
        with path.open("r", encoding="utf-8") as handle:
            return ihm.reader.read(
                handle,
                warn_unknown_category=True,
                warn_unknown_keyword=True,
            )
    except Exception as exc:
        summary["warnings"].append(f"IHM/PDBx parse failed: {exc}")
        return []


def _import_ihm_systems(db: MFDatabase, systems: Iterable[Any], summary: Dict[str, Any]) -> None:
    """Import FLR objects from ``ihm.System`` objects."""
    for system in systems:
        flr_data = getattr(system, "flr_data", None)
        if flr_data is None:
            continue
        _import_ihm_entities(db, system, summary)
        _import_ihm_samples(db, flr_data, summary)
        _import_ihm_probes(db, flr_data, summary)
        _import_ihm_positions(db, flr_data, summary)
        _import_ihm_sample_probes(db, flr_data, summary)
        _import_ihm_analyses(db, flr_data, summary)


def _import_ihm_entities(db: MFDatabase, system: Any, summary: Dict[str, Any]) -> None:
    for entity in getattr(system, "entities", []):
        entity_id = str(getattr(entity, "id", None) or f"entity_{len(summary['entities']) + 1}")
        sequence = [str(res.mon_id) for res in getattr(entity, "sequence", [])]
        db.add_entity(
            entity_id,
            type="polymer",
            description=getattr(entity, "description", None) or entity_id,
            common_name=getattr(entity, "description", None) or entity_id,
        )
        if sequence:
            db.set_sequence(entity_id, sequence)
        summary["entities"].append(entity_id)


def _import_ihm_samples(db: MFDatabase, flr_data: Any, summary: Dict[str, Any]) -> None:
    for sample in getattr(flr_data, "_collection_flr_sample", {}).values():
        sample_id = _object_id(sample) or f"sample_{len(summary['samples']) + 1}"
        condition = getattr(sample, "condition", None)
        condition_id = _object_id(condition) or f"condition_{sample_id}"
        if condition is not None:
            db.add_sample_condition(condition_id, details=getattr(condition, "details", None))
        assembly = getattr(sample, "entity_assembly", None)
        assembly_id = _object_id(assembly) or f"assembly_{sample_id}"
        if assembly is not None:
            db.add_entity_assembly(assembly_id, description=getattr(assembly, "description", None) or assembly_id)
        db.add_sample(
            sample_id,
            description=getattr(sample, "description", None) or sample_id,
            details=getattr(sample, "details", None) or "",
            num_of_probes=getattr(sample, "num_of_probes", None),
            solvent_phase=getattr(sample, "solvent_phase", None),
            sample_condition_id=condition_id,
            entity_assembly_id=assembly_id,
        )
        summary["samples"].append(sample_id)


def _import_ihm_probes(db: MFDatabase, flr_data: Any, summary: Dict[str, Any]) -> None:
    type_id = db.add_probe_type("imported", "Imported probe")
    for probe in getattr(flr_data, "_collection_flr_probe", {}).values():
        entry = getattr(probe, "probe_list_entry", None)
        name = getattr(entry, "chromophore_name", None) or _object_id(probe) or f"probe_{len(db.get_probes()) + 1}"
        existing = None
        for row in db.search_probes(name):
            existing = int(row["probe_id"])
            break
        if existing is None:
            db.add_probe(
                name,
                type_id,
                category="other",
                description=getattr(probe, "description", "") or "",
            )


def _import_ihm_positions(db: MFDatabase, flr_data: Any, summary: Dict[str, Any]) -> None:
    for position in getattr(flr_data, "_collection_flr_poly_probe_position", {}).values():
        resatom = getattr(position, "resatom", None)
        if resatom is None:
            continue
        residue = getattr(resatom, "residue", resatom)
        entity = getattr(residue, "entity", None)
        entity_id = _object_id(entity)
        if not entity_id:
            continue
        seq_id = int(getattr(residue, "seq_id", 1))
        asym_id = str(getattr(residue, "asym", None) or "A")
        probe_id = _first_probe_id(db)
        if probe_id is None:
            continue
        db.add_poly_probe_position(
            probe_id,
            entity_id,
            seq_id,
            asym_id=asym_id,
            residue_name=str(getattr(residue, "mon_id", "")) or None,
            description=getattr(position, "auth_name", None) or getattr(position, "description", None),
        )


def _import_ihm_sample_probes(db: MFDatabase, flr_data: Any, summary: Dict[str, Any]) -> None:
    for spd in getattr(flr_data, "_collection_flr_sample_probe_details", {}).values():
        sample_id = _object_id(getattr(spd, "sample", None))
        probe = getattr(spd, "probe", None)
        probe_id = _probe_id_for(db, probe)
        position = getattr(spd, "poly_probe_position", None)
        position_id = _position_id_for(db, position)
        if sample_id and probe_id:
            db.add_sample_probe(
                sample_id,
                probe_id,
                poly_probe_position_id=position_id,
                fluorophore_type=getattr(spd, "fluorophore_type", "unspecified") or "unspecified",
                description=getattr(spd, "description", "") or "",
            )


def _import_ihm_analyses(db: MFDatabase, flr_data: Any, summary: Dict[str, Any]) -> None:
    for analysis in getattr(flr_data, "_collection_flr_fret_analysis", {}).values():
        analysis_id = _object_id(analysis) or f"analysis_{len(summary['samples']) + 1}"
        sample_probe_1 = getattr(analysis, "sample_probe_1", None)
        sample_probe_2 = getattr(analysis, "sample_probe_2", None)
        sample_probe_id_1 = _sample_probe_id_for(db, sample_probe_1)
        sample_probe_id_2 = _sample_probe_id_for(db, sample_probe_2)
        db.update_analysis_record(
            analysis_id,
            type=getattr(analysis, "type", None) or "intensity-based",
            method=getattr(analysis, "method_name", None),
            sample_id=_object_id(getattr(getattr(analysis, "experiment", None), "sample", None)),
            sample_probe_id_1=sample_probe_id_1,
            sample_probe_id_2=sample_probe_id_2,
            details=getattr(analysis, "details", None),
        )


def _import_chisurf_extensions(db: MFDatabase, path: Path, summary: Dict[str, Any]) -> None:
    """Import ChiSurf extension categories such as spectra and properties."""
    try:
        from pdbx.reader import PdbxReader
        from pdbx import containers
    except Exception as exc:
        summary["warnings"].append(f"pdbx reader unavailable: {exc}")
        return

    data: List[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            PdbxReader(handle).read(data)
    except Exception as exc:
        summary["warnings"].append(f"pdbx extension parse failed: {exc}")
        return

    for block in data:
        for name in block.get_object_name_list():
            category = block.get_object(name)
            rows = _pdbx_category_rows(category)
            category_name = name if name.startswith("_") else f"_{name}"
            if category_name == "_chisurf_probe_property":
                _import_probe_properties(db, rows, summary)
            elif category_name == "_chisurf_probe_spectrum":
                _import_probe_spectra(db, rows, summary)
            elif category_name == "_chisurf_analysis_data":
                _import_analysis_data(db, rows, summary)
            elif category_name == "_struct_ref":
                _import_struct_ref(db, rows, summary)
            elif category_name == "_struct_ref_seq":
                _import_struct_ref_seq(db, rows, summary)
            elif category_name == "_struct_ref_seq_dif":
                _import_struct_ref_seq_dif(db, rows, summary)


def _pdbx_category_rows(category: Any) -> List[Dict[str, Any]]:
    attrs = list(category.attribute_list)
    rows: List[Dict[str, Any]] = []
    for index in range(category.row_count):
        row = {attr: category.get_value(attr, index) for attr in attrs}
        rows.append(row)
    return rows


def _import_probe_properties(db: MFDatabase, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    for row in rows:
        probe_id = _int_or_none(row.get("probe_id"))
        if probe_id is None:
            continue
        if db.get_probe(probe_id) is None:
            type_id = db.add_probe_type("imported", "Imported probe")
            db.add_probe(f"probe_{probe_id}", type_id, category="other")
        db.add_optical_property(
            probe_id,
            str(row.get("property_name") or ""),
            str(row.get("property_value") or ""),
            unit=row.get("unit") or None,
        )


def _import_probe_spectra(db: MFDatabase, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    for row in rows:
        probe_id = _int_or_none(row.get("probe_id"))
        if probe_id is None:
            continue
        if db.get_probe(probe_id) is None:
            type_id = db.add_probe_type("imported", "Imported probe")
            db.add_probe(f"probe_{probe_id}", type_id, category="other")
        wavelengths = _parse_number_list(row.get("wavelengths"))
        values = _parse_number_list(row.get("intensity_values"))
        if not wavelengths or not values:
            continue
        db.add_spectrum(
            probe_id,
            str(row.get("spectrum_type") or "emission"),
            np.asarray(wavelengths, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            wavelength_unit=row.get("wavelength_unit") or "nm",
            intensity_unit=row.get("intensity_unit") or "normalized",
            details=row.get("details") or None,
        )


def _import_analysis_data(db: MFDatabase, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    for row in rows:
        analysis_id = row.get("analysis_id")
        if not analysis_id:
            continue
        x_values = _parse_number_list(row.get("x_values"))
        y_values = _parse_number_list(row.get("y_values"))
        if not x_values or not y_values:
            continue
        db.add_analysis_data(
            str(analysis_id),
            str(row.get("data_type") or "unknown"),
            np.asarray(x_values, dtype=np.float64),
            np.asarray(y_values, dtype=np.float64),
            data_name=row.get("data_name") or None,
            x_unit=row.get("x_unit") or None,
            y_unit=row.get("y_unit") or None,
            details=row.get("details") or None,
        )


def _import_struct_ref(db: MFDatabase, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Import ``_struct_ref`` category rows from FLR CIF."""
    for row in rows:
        ref_id = row.get("ref_id")
        entity_id = row.get("entity_id")
        if not ref_id or not entity_id:
            continue
        db.conn.execute(
            "INSERT OR IGNORE INTO struct_ref "
            "(ref_id, entity_id, db_name, db_code, pdbx_db_accession, "
            "pdbx_db_isoform, pdbx_seq_one_letter_code, organism, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ref_id, entity_id,
             row.get("db_name") or "",
             row.get("db_code"),
             row.get("pdbx_db_accession"),
             row.get("pdbx_db_isoform"),
             row.get("pdbx_seq_one_letter_code"),
             row.get("organism"),
             row.get("details")),
        )
        if entity_id not in summary.setdefault("entities", []):
            summary["entities"].append(entity_id)


def _import_struct_ref_seq(db: MFDatabase, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Import ``_struct_ref_seq`` category rows from FLR CIF."""
    for row in rows:
        align_id = row.get("align_id")
        ref_id = row.get("ref_id")
        if not align_id:
            continue
        db.conn.execute(
            "INSERT OR IGNORE INTO struct_ref_seq "
            "(align_id, ref_id, seq_align_beg, seq_align_end, "
            "db_align_beg, db_align_end, pdbx_db_accession, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (align_id, ref_id,
             _int_or_none(row.get("seq_align_beg")),
             _int_or_none(row.get("seq_align_end")),
             _int_or_none(row.get("db_align_beg")),
             _int_or_none(row.get("db_align_end")),
             row.get("pdbx_db_accession"),
             row.get("details")),
        )


def _import_struct_ref_seq_dif(db: MFDatabase, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Import ``_struct_ref_seq_dif`` category rows from FLR CIF."""
    for row in rows:
        align_id = row.get("align_id")
        seq_num = _int_or_none(row.get("seq_num"))
        if not align_id or seq_num is None:
            continue
        db.conn.execute(
            "INSERT OR IGNORE INTO struct_ref_seq_dif "
            "(align_id, seq_num, mon_id, db_mon_id, details, "
            "pdbx_seq_db_name, pdbx_seq_db_accession_code, pdbx_ordinal) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (align_id, seq_num,
             row.get("mon_id"),
             row.get("db_mon_id"),
             row.get("details"),
             row.get("pdbx_seq_db_name"),
             row.get("pdbx_seq_db_accession_code"),
             _int_or_none(row.get("pdbx_ordinal"))),
        )


def _object_id(obj: Any) -> Optional[str]:
    for attr in ("id", "_id", "name"):
        value = getattr(obj, attr, None)
        if value is not None:
            return str(value)
    return None


def _first_probe_id(db: MFDatabase) -> Optional[int]:
    rows = db.get_probes()
    return int(rows[0]["probe_id"]) if rows else None


def _probe_id_for(db: MFDatabase, probe: Any) -> Optional[int]:
    entry = getattr(probe, "probe_list_entry", None)
    name = getattr(entry, "chromophore_name", None) or _object_id(probe)
    if not name:
        return None
    for row in db.search_probes(name):
        return int(row["probe_id"])
    type_id = db.add_probe_type("imported", "Imported probe")
    return int(db.add_probe(name, type_id, category="other"))


def _position_id_for(db: MFDatabase, position: Any) -> Optional[int]:
    if position is None:
        return None
    resatom = getattr(position, "resatom", None)
    residue = getattr(resatom, "residue", resatom)
    entity_id = _object_id(getattr(residue, "entity", None))
    seq_id = getattr(residue, "seq_id", None)
    if not entity_id or seq_id is None:
        return None
    rows = db.get_poly_probe_positions(entity_id=entity_id)
    for row in rows:
        if int(row["residue_number"]) == int(seq_id):
            return int(row["id"])
    return None


def _sample_probe_id_for(db: MFDatabase, sample_probe: Any) -> Optional[int]:
    if sample_probe is None:
        return None
    sample_id = _object_id(getattr(sample_probe, "sample", None))
    probe_id = _probe_id_for(db, getattr(sample_probe, "probe", None))
    if not sample_id or probe_id is None:
        return None
    rows = db.get_sample_probe_mappings(sample_id=sample_id, probe_id=probe_id)
    return int(rows[0]["sample_probe_id"]) if rows else None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_number_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace(",", " ").split()
    result = []
    for item in raw:
        try:
            result.append(float(item))
        except ValueError:
            continue
    return result


def _sample_id_from_path(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("-", "_")
