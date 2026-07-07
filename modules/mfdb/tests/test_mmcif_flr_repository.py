import pathlib
import tempfile
import numpy as np
import pytest
from mfdb.repository import MFDatabase


@pytest.fixture
def db():
    return MFDatabase(":memory:", enforce_foreign_keys=False)


class TestMFDatabaseProbes:
    def test_add_probe_type(self, db):
        tid = db.add_probe_type("organic_dye", "Organic dye")
        assert isinstance(tid, int)
        row = db.conn.execute("SELECT * FROM probe_types WHERE type_id=?", (tid,)).fetchone()
        assert row is not None
        assert row["type_name"] == "organic_dye"
        assert row["display_name"] == "Organic dye"

    def test_add_probe_type_upsert_preserves_referenced_id(self):
        db = MFDatabase(":memory:")
        try:
            tid = db.add_probe_type("organic_dye", "Organic dye")
            db.add_probe("Alexa488", tid, category="organic_dye")

            updated_tid = db.add_probe_type("organic_dye", "Organic dye updated")

            assert updated_tid == tid
            row = db.conn.execute(
                "SELECT display_name FROM probe_types WHERE type_id = ?",
                (tid,),
            ).fetchone()
            assert row["display_name"] == "Organic dye updated"
        finally:
            db.close()

    def test_add_probe(self, db):
        tid = db.add_probe_type("organic_dye", "Organic dye")
        probe_id = db.add_probe("Alexa488", tid, category="organic_dye")
        assert isinstance(probe_id, int)
        row = db.conn.execute("SELECT * FROM probes WHERE probe_id=?", (probe_id,)).fetchone()
        assert row is not None
        assert row["chromophore_name"] == "Alexa488"

    def test_add_spectrum(self, db):
        tid = db.add_probe_type("organic_dye", "Organic dye")
        probe_id = db.add_probe("Cy5", tid, category="organic_dye")
        x = np.array([600.0, 620.0, 640.0])
        y = np.array([0.1, 0.9, 0.3])
        db.add_spectrum(probe_id, "emission", x, y, wavelength_unit="nm")
        row = db.conn.execute(
            "SELECT * FROM spectra WHERE probe_id=? AND spectrum_type='emission'",
            (probe_id,),
        ).fetchone()
        assert row is not None
        assert row["wavelength_unit"] == "nm"

    def test_add_spectrum_wavelength_unit_default(self, db):
        tid = db.add_probe_type("organic_dye", "Organic dye")
        probe_id = db.add_probe("Cy5", tid, category="organic_dye")
        x = np.array([600.0, 620.0, 640.0])
        y = np.array([0.1, 0.9, 0.3])
        db.add_spectrum(probe_id, "emission", x, y)
        row = db.conn.execute(
            "SELECT * FROM spectra WHERE probe_id=? AND spectrum_type='emission'",
            (probe_id,),
        ).fetchone()
        assert row is not None
        assert row["wavelength_unit"] == "nm"

    def test_get_spectrum_record(self, db):
        tid = db.add_probe_type("organic_dye", "Organic dye")
        probe_id = db.add_probe("Cy5", tid, category="organic_dye")
        x = np.array([600.0, 620.0, 640.0])
        y = np.array([0.1, 0.9, 0.3])
        db.add_spectrum(probe_id, "emission", x, y)
        record = db.get_spectrum_record(probe_id, "emission")
        assert record is not None
        assert record["probe_id"] == probe_id
        assert record["spectrum_type"] == "emission"

    def test_get_spectrum_record_missing(self, db):
        record = db.get_spectrum_record(999999, "emission")
        assert record is None

    def test_add_optical_property(self, db):
        tid = db.add_probe_type("organic_dye", "Organic dye")
        probe_id = db.add_probe("Alexa488", tid, category="organic_dye")
        db.add_optical_property(probe_id, "qy", "0.92", unit="")
        row = db.conn.execute(
            "SELECT * FROM optical_properties WHERE probe_id=? AND property_name='qy'",
            (probe_id,),
        ).fetchone()
        assert row is not None
        assert row["property_value"] == "0.92"


class TestMFDatabaseAnalysis:
    def test_update_analysis_record(self, db):
        db.update_analysis_record("analysis_1", sample_id="sample_1", type="intensity-based")
        row = db.conn.execute(
            "SELECT * FROM flr_fret_analysis WHERE analysis_id='analysis_1'"
        ).fetchone()
        assert row is not None
        assert row["sample_id"] == "sample_1"
        assert row["type"] == "intensity-based"

    def test_analysis_metadata(self, db):
        db.update_analysis_record("analysis_2", sample_id="sample_2")
        db.set_analysis_metadata("analysis_2", {"pH": "7.4", "temperature": "298 K"})
        meta = db.get_analysis_metadata("analysis_2")
        assert meta == {"pH": "7.4", "temperature": "298 K"}

    def test_delete_analysis_metadata(self, db):
        db.update_analysis_record("analysis_3", sample_id="sample_3")
        db.set_analysis_metadata("analysis_3", {"key1": "val1", "key2": "val2"})
        db.delete_analysis_metadata("analysis_3", "key1")
        meta = db.get_analysis_metadata("analysis_3")
        assert "key1" not in meta
        assert meta["key2"] == "val2"

    def test_add_analysis_metadata(self, db):
        db.update_analysis_record("analysis_4", sample_id="sample_4")
        db.add_analysis_metadata("analysis_4", "key1", "value1")
        db.add_analysis_metadata("analysis_4", "key2", "value2")
        meta = db.get_analysis_metadata("analysis_4")
        assert meta["key1"] == "value1"
        assert meta["key2"] == "value2"

    def test_add_analysis_metadata_duplicate_key_replaces(self, db):
        db.update_analysis_record("analysis_5", sample_id="sample_5")
        db.add_analysis_metadata("analysis_5", "key", "old")
        db.add_analysis_metadata("analysis_5", "key", "new")
        meta = db.get_analysis_metadata("analysis_5")
        assert meta["key"] == "new"


class TestMFDatabaseExternalFiles:
    def test_add_external_file(self, db):
        with tempfile.TemporaryDirectory() as d:
            fpath = pathlib.Path(d) / "test.ptu"
            fpath.write_text("fake ptu data")
            file_id = db.add_external_file(fpath, file_format="ptu")
            assert isinstance(file_id, int)
            row = db.conn.execute(
                "SELECT * FROM ihm_external_files WHERE id=?", (file_id,)
            ).fetchone()
            assert row is not None
            assert row["file_path"] == str(fpath)
            assert row["file_format"] == "ptu"

    def test_get_external_file(self, db):
        with tempfile.TemporaryDirectory() as d:
            fpath = pathlib.Path(d) / "test.ptu"
            fpath.write_text("fake ptu data")
            file_id = db.add_external_file(fpath, file_format="ptu")
            row = db.get_external_file(file_id)
            assert row is not None
            assert row["id"] == file_id

    def test_add_photon_stream(self, db):
        with tempfile.TemporaryDirectory() as d:
            db.update_analysis_record("analysis_ps", sample_id="sample_ps")
            fpath = pathlib.Path(d) / "photons.ptu"
            fpath.write_text("fake photon data")
            db.add_photon_stream("analysis_ps", fpath, file_format="ptu", detector_id="det1")
            streams = db.get_photon_streams("analysis_ps")
            assert len(streams) == 1
            row = streams[0]
            assert row["analysis_id"] == "analysis_ps"
            assert row["detector_id"] == "det1"

    def test_add_photon_stream_multiple(self, db):
        with tempfile.TemporaryDirectory() as d:
            db.update_analysis_record("analysis_ps2", sample_id="sample_ps2")
            f1 = pathlib.Path(d) / "photons1.ptu"
            f2 = pathlib.Path(d) / "photons2.ptu"
            f1.write_text("data1")
            f2.write_text("data2")
            db.add_photon_stream("analysis_ps2", f1, file_format="ptu", detector_id="det1")
            db.add_photon_stream("analysis_ps2", f2, file_format="ptu", detector_id="det2")
            streams = db.get_photon_streams("analysis_ps2")
            assert len(streams) == 2


class TestMFDatabaseExport:
    def test_export_flr_cif_basic(self, db):
        with tempfile.TemporaryDirectory() as d:
            tid = db.add_probe_type("organic_dye", "Organic dye")
            probe_id = db.add_probe("mCherry", tid, category="organic_dye")
            db.add_spectrum(
                probe_id,
                "emission",
                np.array([580.0, 600.0, 620.0]),
                np.array([0.2, 1.0, 0.4]),
            )
            db.add_optical_property(probe_id, "qy", "0.22", unit="")
            db.update_analysis_record(
                "analysis_export", sample_id="sample_export", type="intensity-based"
            )
            db.set_analysis_metadata("analysis_export", {"buffer": "PBS"})
            fpath = pathlib.Path(d) / "photons.ptu"
            fpath.write_text("fake")
            db.add_photon_stream("analysis_export", fpath, file_format="ptu", detector_id="det1")
            out = pathlib.Path(d) / "export.cif"
            db.export_flr_cif(out, analysis_id="analysis_export")
            text = out.read_text()
            assert "_chisurf_probe_spectrum" in text
            assert "_chisurf_photon_stream" in text
            assert "mCherry" in text

    def test_export_flr_cif_fallback_analysis_id(self, db):
        """Export works without specifying analysis_id (auto-fallback to first)."""
        with tempfile.TemporaryDirectory() as d:
            db.update_analysis_record("test_analysis", sample_id="test_s")
            out = pathlib.Path(d) / "export.cif"
            db.export_flr_cif(out)  # no analysis_id given
            text = out.read_text()
            assert "_flr_fret_analysis" in text

    def test_export_flr_cif_embeds_analysis_data(self, db):
        db.update_analysis_record("analysis_data", sample_id="sample_data")
        db.add_analysis_data(
            "analysis_data",
            "decay",
            np.array([0.0, 1.0, 2.0]),
            np.array([100.0, 50.0, 25.0]),
            data_name="donor_decay",
            x_unit="ns",
            y_unit="counts",
            details="TCSPC decay",
        )
        db.add_analysis_data(
            "analysis_data",
            "decay",
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([100.0, 50.0, 25.0, 12.5]),
            data_name="donor_decay",
            x_unit="ns",
            y_unit="counts",
            details="TCSPC decay",
        )
        assert len([row for row in db.get_analysis_data("analysis_data") if row["data_name"] == "donor_decay"]) == 1
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "analysis_data.cif"
            db.export_flr_cif(out, analysis_id="analysis_data")
            text = out.read_text()
            assert "_chisurf_analysis_data" in text
            assert "decay" in text
            assert "donor_decay" in text
            assert "0 1 2 3" in text
            assert "100 50 25 12.5" in text


class TestMFDatabaseSamples:
    def test_sample_probe_mapping_crud(self, db):
        db.add_sample("sample_1", uuid="uuid-1")
        probe_id = db.add_probe(
            "Alexa488", db.add_probe_type("organic_dye", "Organic dye"), category="organic_dye"
        )
        sample_probe_id = db.add_sample_probe(
            "sample_1", probe_id, fluorophore_type="donor", description="donor site"
        )
        mappings = db.get_sample_probe_mappings(sample_id="sample_1")
        assert len(mappings) == 1
        assert mappings[0]["sample_probe_id"] == sample_probe_id
        assert mappings[0]["chromophore_name"] == "Alexa488"
        assert db.get_sample_probe_mappings()[0]["sample_probe_id"] == sample_probe_id
        assert db.get_sample_probe_mappings(probe_id=probe_id)[0]["sample_probe_id"] == sample_probe_id
        db.clear_sample_probes("sample_1")
        assert db.get_sample_probe_mappings(sample_id="sample_1") == []

    def test_sample_lims_metadata_crud(self, db):
        db.add_user("operator", "Demo Operator")
        db.add_device("tcspc", "Demo TCSPC", location="Lab 1")
        db.add_sample(
            "sample_lims",
            measured_by_user_id="operator",
            measured_by_device_id="tcspc",
            project_id="demo",
            measured_at="2026-06-11T09:00:00",
        )
        db.set_sample_key_value("sample_lims", "pdbx.sample_type", "protein")
        sample = db.get_sample_full("sample_lims")
        assert sample["project_id"] == "demo"
        assert sample["measured_by_user_id"] == "operator"
        assert sample["measured_by_device_id"] == "tcspc"
        assert sample["key_values"][0]["key"] == "pdbx.sample_type"
        assert db.get_sample_key_values("sample_lims")[0]["value"] == "protein"

    def test_export_uses_sample_probe_mapping(self, db):
        donor_type = db.add_probe_type("organic_dye", "Organic dye")
        donor = db.add_probe("Alexa488", donor_type, category="organic_dye")
        acceptor = db.add_probe("Alexa594", donor_type, category="organic_dye")
        db.add_sample(
            "sample_export_mapping", uuid="uuid-sample", num_of_probes=2, solvent_phase="liquid"
        )
        donor_sp = db.add_sample_probe("sample_export_mapping", donor, fluorophore_type="donor")
        db.add_sample_probe("sample_export_mapping", acceptor, fluorophore_type="acceptor")
        db.update_analysis_record(
            "analysis_mapping",
            sample_id="sample_export_mapping",
            type="intensity-based",
            sample_probe_id_1=donor_sp,
            sample_probe_id_2=donor_sp + 1,
        )
        import pathlib, tempfile

        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "mapping.cif"
            db.export_flr_cif(out, analysis_id="analysis_mapping")
            text = out.read_text()
            assert "sample_export_mapping" in text
            assert "_flr_sample_probe_details.sample_probe_id" in text
            assert "1 sample_export_mapping 1" in text


class TestMFDatabaseExperiments:
    def test_experiment_type_crud(self, db):
        type_id = db.add_experiment_type("tcspc", category="Time-resolved")
        assert isinstance(type_id, int)
        rows = db.get_experiment_types()
        assert any(row["name"] == "tcspc" and row["category"] == "Time-resolved" for row in rows)
        db.delete_experiment_type(type_id)
        assert db.get_experiment_types() == []

    def test_experiment_lims_crud(self, db):
        type_id = db.add_experiment_type("tcspc", category="Time-resolved")
        db.add_user("operator", "Demo Operator")
        db.add_device("tcspc_device", "Demo TCSPC")
        db.add_sample("sample_exp")
        db.add_experiment(
            "experiment_1",
            type_id=type_id,
            sample_id="sample_exp",
            project_id="demo",
            measured_by_user_id="operator",
            measured_by_device_id="tcspc_device",
            started_at="2026-06-11T09:00:00",
            status="completed",
        )
        db.set_experiment_key_value("experiment_1", "flrcif.experiment_type", "tcspc")
        data_id = db.add_experiment_data(
            "experiment_1",
            data_type="tcspc",
            storage_mode="link",
            file_path="raw/demo.ptu",
            mime_type="application/octet-stream",
            checksum="demo",
            reading_options_json='{"skiprows":0}',
        )
        full = db.get_experiment_full("experiment_1")
        assert full["experiment_type"] == "tcspc"
        assert full["measured_by_user"] == "Demo Operator"
        assert full["key_values"][0]["key"] == "flrcif.experiment_type"
        assert full["data"][0]["data_id"] == data_id
        assert full["data"][0]["reading_options_json"] == '{"skiprows":0}'
        assert len(db.get_experiments(sample_id="sample_exp")) == 1
        db.delete_experiment_data(data_id)
        assert db.get_experiment_data("experiment_1") == []
        db.delete_experiment("experiment_1")
        assert db.get_experiment("experiment_1") is None

    # Note: the former test_experiment_id_column_repair drove the removed version-chain
    # migration (`_schema_version=12` + a v12-era column-repair step renaming
    # flr_experiment.id back to experiment_id). PRD-19 deleted the version chain
    # (pre-PRD-19 DBs are disposable); a fresh DB has experiment_id by construction.


class TestMFDatabaseMigration:
    def test_empty_db_schema(self, db):
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in tables}
        assert "probes" in names
        assert "spectra" in names
        assert "optical_properties" in names
        assert "flr_fret_analysis" in names
        assert "analysis_metadata" in names
        assert "flr_photon_stream" in names
        assert "ihm_external_files" in names
