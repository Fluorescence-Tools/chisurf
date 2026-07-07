"""Tests for PDBx/PDB-IHM/FLR CIF import support."""

from __future__ import annotations

import tempfile
from pathlib import Path

from mfdb.samples.importer import import_structure_file
from mfdb.repository import MFDatabase


def test_import_flr_cif_extension_categories():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.cif"
        path.write_text(
            "data_test\n"
            "_flr_probe_list.probe_id 1\n"
            "_flr_probe_list.chromophore_name Alexa488\n"
            "_flr_probe_list.reactive_probe_flag no\n"
            "_flr_probe_list.reactive_probe_name ?\n"
            "_flr_probe_list.probe_origin extrinsic\n"
            "_flr_probe_list.probe_link_type covalent\n"
            "_flr_sample.id sample_import\n"
            "_flr_sample.entity_assembly_id assembly_import\n"
            "_flr_sample.num_of_probes 1\n"
            "_flr_sample.sample_condition_id condition_import\n"
            "_flr_sample.sample_description Imported\n"
            "_flr_sample.sample_details Details\n"
            "_flr_sample.solvent_phase liquid\n"
            "_flr_sample_condition.id condition_import\n"
            "_flr_sample_condition.details pH=7\n"
            "_flr_sample_probe_details.sample_probe_id 1\n"
            "_flr_sample_probe_details.sample_id sample_import\n"
            "_flr_sample_probe_details.probe_id 1\n"
            "_flr_sample_probe_details.fluorophore_type donor\n"
            "_flr_sample_probe_details.description donor\n"
            "_flr_sample_probe_details.poly_probe_position_id ?\n"
            "_chisurf_probe_property.probe_id 1\n"
            "_chisurf_probe_property.property_name abs_max\n"
            "_chisurf_probe_property.property_value 495\n"
            "_chisurf_probe_property.unit nm\n"
            "_chisurf_probe_property.details seed\n"
            "_chisurf_probe_spectrum.probe_id 1\n"
            "_chisurf_probe_spectrum.spectrum_type absorption\n"
            '_chisurf_probe_spectrum.wavelengths "455 495 535"\n'
            '_chisurf_probe_spectrum.intensity_values "0.2 1.0 0.2"\n'
            "_chisurf_probe_spectrum.wavelength_unit nm\n"
            "_chisurf_probe_spectrum.intensity_unit normalized\n"
            "_chisurf_probe_spectrum.details spectrum\n"
        )
        db = MFDatabase(":memory:")
        try:
            summary = import_structure_file(db, path)
            assert summary["samples"]
            sample_id = summary["samples"][0]
            assert db.get_sample(sample_id) is not None
            props = {
                p["property_name"]: p["property_value"]
                for p in db.get_optical_properties(1)
            }
            assert props["abs_max"] == "495"
            assert db.get_spectrum_record(1, "absorption") is not None
        finally:
            db.close()
