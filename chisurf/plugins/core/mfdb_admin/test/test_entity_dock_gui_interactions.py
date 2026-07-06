"""Headless MFDB Admin EntityDock interaction tests with real sample data."""
from __future__ import annotations

import os
from contextlib import contextmanager
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from mfdb.admin.gui.client import MFDBClient

from .conftest import patch_db


_WIDGETS: list = []


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def seeded_admin_db(db, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mfdb.database_resolver.object_store_root",
        lambda: tmp_path / "objects",
    )
    db.add_user("user_default", "Default User")
    db.add_device("dev_gui", "GUI Detector")
    db.add_sample_condition(
        "cond_gui",
        ph=7.4,
        temperature=298.0,
        buffer_composition="PBS",
    )
    type_id = db.add_experiment_type("TCSPC", category="fluorescence")
    db.add_sample(
        "sample_gui",
        description="GUI sample",
        sample_condition_id="cond_gui",
        measured_by_user_id="user_default",
        measured_by_device_id="dev_gui",
    )
    db.set_sample_key_value("sample_gui", "buffer", "PBS", "initial buffer")
    db.add_entity(
        "entity_gui",
        name="GUI Entity",
        sequence=["ALA", "CYS", "GLY"],
        entity_type="polymer",
        details="GUI entity details",
    )
    donor_probe_id = db.add_probe(
        None,
        name="Donor GUI",
        category="organic_dye",
        description="initial donor",
        fluorophore_type="donor",
    )
    acceptor_probe_id = db.add_probe(
        None,
        name="Acceptor GUI",
        category="organic_dye",
        description="initial acceptor",
        fluorophore_type="acceptor",
    )
    position_id = db.add_poly_probe_position(
        donor_probe_id,
        "entity_gui",
        2,
        asym_id="A",
        residue_name="CYS",
        atom_id="CB",
        auth_name="C2",
        description="GUI label position",
    )
    db.add_sample_probe(
        "sample_gui",
        donor_probe_id,
        "donor",
        poly_probe_position_id=position_id,
    )
    db.add_sample_probe("sample_gui", acceptor_probe_id, "acceptor")
    db.add_fret_forster_radius(
        "fr_gui",
        "sample_gui",
        donor_probe_id,
        acceptor_probe_id,
        6.1,
        details="GUI FRET pair",
    )
    db.add_experiment(
        "exp_gui",
        type_id=type_id,
        sample_id="sample_gui",
        measured_by_user_id="user_default",
        measured_by_device_id="dev_gui",
        status="complete",
    )
    db.save_setup(
        setup_id="setup_gui",
        name="GUI Setup",
        configuration={"setup_type": "confocal", "details": "initial setup"},
        detectors={
            "green": {
                "channels": [0, 1],
                "micro_time_ranges": [[10, 90]],
                "g_factor": 1.05,
                "l1": 0.01,
                "l2": 0.02,
            }
        },
        windows={"donor_excitation": [10, 90]},
        fcs_pairs={
            "green_auto": {
                "channel_a": "green",
                "channel_b": "green",
                "kind": "auto",
                "n_bins": 8,
                "n_casc": 16,
                "make_fine": True,
            }
        },
        created_by_user_id="user_default",
        is_public=False,
    )
    raw_path = tmp_path / "raw_gui.ptu"
    raw_path.write_bytes(b"PTU seed data")
    processed_path = tmp_path / "prod_gui.json"
    processed_path.write_text('{"curve": [1, 2, 3]}', encoding="utf-8")
    db.register_artifact(
        artifact_id="raw_gui",
        artifact_kind="raw_data",
        storage_mode="local_file",
        experiment_id="exp_gui",
        file_path=str(raw_path),
        validation_status="valid",
        metadata={"data_type": "tcspc"},
    )
    db.record_operation(
        operation_id="proc_gui",
        operation_type="filtering",
        experiment_id="exp_gui",
        operator_user_id="user_default",
        status="succeeded",
        settings={"threshold": 3},
    )
    db.register_artifact(
        artifact_id="prod_gui",
        artifact_kind="processed_data",
        storage_mode="local_file",
        experiment_id="exp_gui",
        file_path=str(processed_path),
        validation_status="valid",
        metadata={"processing_id": "proc_gui", "product_type": "processed_data"},
    )
    db.record_operation_link(
        operation_id="proc_gui",
        artifact_id="raw_gui",
        direction="input",
    )
    db.record_operation_link(
        operation_id="proc_gui",
        artifact_id="prod_gui",
        direction="output",
    )
    db.record_operation(
        operation_id="analysis_gui",
        operation_type="local_fit",
        experiment_id="exp_gui",
        operator_user_id="user_default",
        status="succeeded",
        metadata={"model_name": "GUI Fit"},
    )
    analysis_product_path = tmp_path / "analysis_fit_result.json"
    analysis_product_path.write_text('{"chi2": 1.05}', encoding="utf-8")
    db.add_analysis_parameter(
        "analysis_gui",
        "tau",
        value=3.8,
        standard_error=0.2,
        units="ns",
        parameter_type="free",
        parameter_uuid="param_gui_tau",
    )
    db.add_analysis_product(
        "analysis_gui",
        product_type="processed_data",
        storage_mode="local_file",
        processed_data_id="fit_result_gui",
        file_path=str(analysis_product_path),
        validation_status="valid",
        metadata={"product_type": "fit_result"},
    )
    db.put_object(data=b"time,intensity\n0,10\n", filename="sample_curve.csv")
    db.record_operation(
        operation_id="ver_gui",
        operation_type="project",
        operator_user_id="user_default",
        status="succeeded",
        metadata={
            "project_id": "proj_gui",
            "project_name": "GUI Project",
            "version_number": 1,
            "description": "Project from GUI test",
        },
    )
    db.create_branch(
        branch_uuid="branch_gui",
        name="gui_branch",
        head_operation_id="ver_gui",
        created_by_user_id="user_default",
        description="GUI branch",
    )
    return db


@contextmanager
def _widget_for_db(db):
    from mfdb.admin.gui.tool import MFDBWidget

    with patch_db(db):
        client = MFDBClient(inprocess=True)
        with mock.patch.object(MFDBWidget, "_verify_admin_access", lambda s: None), \
             mock.patch.object(MFDBWidget, "_ensure_authenticated", lambda s: None):
            widget = MFDBWidget(client=client)
        _WIDGETS.append(widget)
        yield widget


def _entity_dock(widget, qapp, entity_key):
    row = widget._row_by_entity[entity_key]
    widget.nav_list.setCurrentRow(row)
    qapp.processEvents()
    dock = widget._entity_docks[entity_key]
    if hasattr(dock, "refresh"):
        dock.refresh()
    elif hasattr(dock, "refresh_samples"):
        dock.refresh_samples()
    qapp.processEvents()
    return dock


def _select_row(dock, qapp, row_id):
    dock.select_row_by_id(row_id)
    qapp.processEvents()
    assert dock.selected_row_id() == row_id
    data = dock.form.get_data()
    assert data
    return data


def _select_panel(widget, qapp, panel_name):
    row = widget._row_by_name[panel_name]
    widget.nav_list.setCurrentRow(row)
    qapp.processEvents()


def test_sample_entity_dock_selects_and_auto_saves_sample(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "sample")

        data = _select_row(dock, qapp, "sample_gui")
        assert data["description"] == "GUI sample"

        dock.form._model.description = "GUI sample updated"
        qapp.processEvents()

    assert seeded_admin_db.get_sample("sample_gui")["description"] == "GUI sample updated"


def test_entity_probe_position_and_fret_pair_docks_browse_and_save_seeded_data(
    seeded_admin_db,
    qapp,
):
    with _widget_for_db(seeded_admin_db) as widget:
        entity_dock = _entity_dock(widget, qapp, "entity")
        entity_data = _select_row(entity_dock, qapp, "entity_gui")
        assert entity_data["entity_id"] == "entity_gui"

        probe_dock = _entity_dock(widget, qapp, "probe")
        donor_id = str(
            seeded_admin_db.conn.execute(
                "SELECT probe_id FROM probes WHERE chromophore_name = ?",
                ("Donor GUI",),
            ).fetchone()["probe_id"]
        )
        probe_data = _select_row(probe_dock, qapp, donor_id)
        assert probe_data["chromophore_name"] == "Donor GUI"
        probe_dock.form._model.description = "GUI donor updated"
        qapp.processEvents()

        position_id = str(
            seeded_admin_db.conn.execute(
                "SELECT id FROM flr_poly_probe_position WHERE auth_name = ?",
                ("C2",),
            ).fetchone()["id"]
        )
        position_dock = _entity_dock(widget, qapp, "position")
        position_data = _select_row(position_dock, qapp, position_id)
        assert position_data["entity_id"] == "entity_gui"
        assert position_data["auth_name"] == "C2"

        fret_pair_dock = _entity_dock(widget, qapp, "fret_pair")
        pair_data = _select_row(fret_pair_dock, qapp, "fr_gui")
        assert pair_data["sample_id"] == "sample_gui"
        assert str(pair_data["donor_probe_id"]) == donor_id

    donor = seeded_admin_db.conn.execute(
        "SELECT description FROM probes WHERE chromophore_name = ?",
        ("Donor GUI",),
    ).fetchone()
    assert donor["description"] == "GUI donor updated"


def test_sample_metadata_dock_loads_edits_and_saves_metadata(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "metadata")
        dock.load_sample("sample_gui")
        qapp.processEvents()

        assert dock._editor.table.rowCount() == 1
        dock._editor.table.selectRow(0)
        qapp.processEvents()
        dock._detail_key.setEditText("buffer")
        dock._detail_value.setText("HEPES")
        dock._detail_details.setText("edited buffer")
        dock._apply_detail_to_row()
        dock._save_metadata()
        qapp.processEvents()

    values = {
        row["key"]: row for row in seeded_admin_db.get_sample_key_values("sample_gui")
    }
    assert values["buffer"]["value"] == "HEPES"
    assert values["buffer"]["details"] == "edited buffer"


def test_experiment_entity_dock_selects_and_auto_saves_status(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "experiment")

        data = _select_row(dock, qapp, "exp_gui")
        assert data["sample_id"] == "sample_gui"

        dock.form._model.status = "reviewed"
        qapp.processEvents()

    assert dict(seeded_admin_db.get_experiment("exp_gui"))["status"] == "reviewed"


def test_condition_device_experiment_type_and_setup_docks_auto_save(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        condition_dock = _entity_dock(widget, qapp, "condition")
        condition_data = _select_row(condition_dock, qapp, "cond_gui")
        assert condition_data["buffer_composition"] == "PBS"
        condition_dock.form._model.ph = 7.8
        qapp.processEvents()

        device_dock = _entity_dock(widget, qapp, "device")
        device_data = _select_row(device_dock, qapp, "dev_gui")
        assert device_data["name"] == "GUI Detector"
        device_dock.form._model.model = "GUI-Model"
        qapp.processEvents()

        type_id = str(seeded_admin_db.get_experiment_types()[0]["type_id"])
        experiment_type_dock = _entity_dock(widget, qapp, "experiment_type")
        experiment_type_data = _select_row(experiment_type_dock, qapp, type_id)
        assert experiment_type_data["name"] == "TCSPC"
        experiment_type_dock.form._model.description = "GUI experiment type"
        qapp.processEvents()

        setup_dock = _entity_dock(widget, qapp, "setup")
        setup_data = _select_row(setup_dock, qapp, "setup_gui")
        assert setup_data["name"] == "GUI Setup"
        setup_dock.form._model.details = "updated setup"
        qapp.processEvents()

    condition = seeded_admin_db.conn.execute(
        "SELECT ph FROM flr_sample_condition WHERE condition_id = ?",
        ("cond_gui",),
    ).fetchone()
    assert condition["ph"] == 7.8
    assert seeded_admin_db.get_devices()[0]["model"] == "GUI-Model"
    assert seeded_admin_db.get_experiment_types()[0]["description"] == "GUI experiment type"
    setup = seeded_admin_db.get_setup("setup_gui")
    assert "updated setup" in setup["configuration_json"]


def test_condition_and_device_docks_create_update_and_delete_records(
    seeded_admin_db,
    qapp,
):
    from qtpy import QtCore, QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        condition_dock = _entity_dock(widget, qapp, "condition")
        condition_count = condition_dock.table.rowCount()
        condition_dock._on_new()
        qapp.processEvents()
        condition_id = condition_dock.selected_row_id()
        assert condition_id
        condition_dock.form._model.ph = 6.9
        qapp.processEvents()
        condition_row = seeded_admin_db.conn.execute(
            "SELECT ph FROM flr_sample_condition WHERE condition_id = ? AND deleted_at IS NULL",
            (condition_id,),
        ).fetchone()
        assert condition_row["ph"] == 6.9
        assert condition_dock.table.rowCount() == condition_count + 1

        checkbox = condition_dock.table.item(condition_dock.table.currentRow(), 0)
        checkbox.setCheckState(QtCore.Qt.Checked)
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            condition_dock._on_delete()
        qapp.processEvents()
        assert seeded_admin_db.conn.execute(
            "SELECT 1 FROM flr_sample_condition WHERE condition_id = ? AND deleted_at IS NULL",
            (condition_id,),
        ).fetchone() is None

        device_dock = _entity_dock(widget, qapp, "device")
        device_count = device_dock.table.rowCount()
        device_dock._on_new()
        qapp.processEvents()
        device_id = device_dock.selected_row_id()
        assert device_id
        device_dock.form._model.name = "GUI Created Device"
        qapp.processEvents()
        device_row = seeded_admin_db.conn.execute(
            "SELECT name FROM flr_sample_devices WHERE device_id = ? AND deleted_at IS NULL",
            (device_id,),
        ).fetchone()
        assert device_row["name"] == "GUI Created Device"
        assert device_dock.table.rowCount() == device_count + 1

        checkbox = device_dock.table.item(device_dock.table.currentRow(), 0)
        checkbox.setCheckState(QtCore.Qt.Checked)
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            device_dock._on_delete()
        qapp.processEvents()
        assert seeded_admin_db.conn.execute(
            "SELECT 1 FROM flr_sample_devices WHERE device_id = ? AND deleted_at IS NULL",
            (device_id,),
        ).fetchone() is None


def test_experiment_type_dock_creates_updates_and_deletes_type(
    seeded_admin_db,
    qapp,
):
    from qtpy import QtCore, QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "experiment_type")
        original_count = dock.table.rowCount()

        dock._on_new()
        qapp.processEvents()

        type_id = dock.selected_row_id()
        assert type_id
        created = seeded_admin_db.conn.execute(
            "SELECT * FROM flr_experiment_type WHERE type_id = ? AND deleted_at IS NULL",
            (int(type_id),),
        ).fetchone()
        assert created is not None
        assert created["name"].startswith("untitled_experiment_type_")
        assert dock.table.rowCount() == original_count + 1

        dock.form._model.description = "created and updated from GUI"
        qapp.processEvents()
        updated = seeded_admin_db.conn.execute(
            "SELECT description FROM flr_experiment_type WHERE type_id = ? AND deleted_at IS NULL",
            (int(type_id),),
        ).fetchone()
        assert updated["description"] == "created and updated from GUI"

        checkbox = dock.table.item(dock.table.currentRow(), 0)
        checkbox.setCheckState(QtCore.Qt.Checked)
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            dock._on_delete()
        qapp.processEvents()

    assert seeded_admin_db.conn.execute(
        "SELECT 1 FROM flr_experiment_type WHERE type_id = ? AND deleted_at IS NULL",
        (int(type_id),),
    ).fetchone() is None


def test_setup_child_docks_browse_seeded_detector_pie_and_fcs_rows(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        detector_dock = _entity_dock(widget, qapp, "detector_channel")
        assert detector_dock.table.rowCount() == 1
        detector_id = detector_dock.table.item(0, 1).text()
        detector_data = _select_row(detector_dock, qapp, detector_id)
        assert detector_data["setup_id"] == "setup_gui"
        assert detector_data["name"] == "green"
        assert "0" in str(detector_data["channels"])

        pie_dock = _entity_dock(widget, qapp, "pie_window")
        assert pie_dock.table.rowCount() == 1
        pie_id = pie_dock.table.item(0, 1).text()
        pie_data = _select_row(pie_dock, qapp, pie_id)
        assert pie_data["setup_id"] == "setup_gui"
        assert pie_data["name"] == "donor_excitation"
        assert int(pie_data["start"]) == 10
        assert int(pie_data["end"]) == 90

        fcs_dock = _entity_dock(widget, qapp, "fcs_pair")
        assert fcs_dock.table.rowCount() == 1
        fcs_id = fcs_dock.table.item(0, 1).text()
        fcs_data = _select_row(fcs_dock, qapp, fcs_id)
        assert fcs_data["setup_id"] == "setup_gui"
        assert fcs_data["name"] == "green_auto"
        assert fcs_data["channel_a"] == "green"
        assert fcs_data["channel_b"] == "green"


def test_user_entity_dock_selects_and_auto_saves_display_name(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "user")

        data = _select_row(dock, qapp, "user_default")
        assert data["display_name"] == "Default User"

        dock.form._model.display_name = "GUI Default User"
        qapp.processEvents()

    users = {row["user_id"]: row for row in seeded_admin_db.get_users()}
    assert users["user_default"]["display_name"] == "GUI Default User"


def test_user_entity_dock_creates_renames_and_deletes_user(seeded_admin_db, qapp):
    from qtpy import QtCore, QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "user")
        original_count = dock.table.rowCount()

        dock._on_new()
        qapp.processEvents()

        created_id = dock.selected_row_id()
        assert created_id
        users = {row["user_id"]: row for row in seeded_admin_db.get_users()}
        assert created_id in users
        assert users[created_id]["display_name"] == created_id

        dock.form._model.display_name = "GUI Created User"
        qapp.processEvents()
        dock.form._model.user_id = "gui_created_user"
        qapp.processEvents()

        users = {row["user_id"]: row for row in seeded_admin_db.get_users()}
        assert created_id not in users
        assert users["gui_created_user"]["display_name"] == "GUI Created User"
        assert dock.table.rowCount() == original_count + 1

        checkbox = dock.table.item(dock.table.currentRow(), 0)
        checkbox.setCheckState(QtCore.Qt.Checked)
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            dock._on_delete()
        qapp.processEvents()

    users = {row["user_id"]: row for row in seeded_admin_db.get_users()}
    assert "gui_created_user" not in users


def test_user_entity_dock_prevents_builtin_user_delete(seeded_admin_db, qapp):
    from qtpy import QtCore, QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "user")
        _select_row(dock, qapp, "user_default")
        checkbox = dock.table.item(dock.table.currentRow(), 0)
        checkbox.setCheckState(QtCore.Qt.Checked)

        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ), mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            dock._on_delete()
        qapp.processEvents()

        warning.assert_called_once()
        assert "built-in user" in warning.call_args.args[2]

    users = {row["user_id"]: row for row in seeded_admin_db.get_users()}
    assert "user_default" in users


def test_object_project_and_branch_entity_docks_browse_seeded_data(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        object_dock = _entity_dock(widget, qapp, "object")
        assert object_dock.table.rowCount() == 1
        object_id = object_dock.table.item(0, 1).text()
        object_data = _select_row(object_dock, qapp, object_id)
        assert object_data["original_filename"] == "sample_curve.csv"

        project_dock = _entity_dock(widget, qapp, "project")
        project_data = _select_row(project_dock, qapp, "proj_gui")
        assert project_data["name"] == "GUI Project"

        branch_dock = _entity_dock(widget, qapp, "branch")
        branch_data = _select_row(branch_dock, qapp, "branch_gui")
        assert branch_data["name"] == "gui_branch"


def test_object_entity_dock_copies_reveals_and_deletes_seeded_object(
    seeded_admin_db,
    qapp,
):
    from qtpy import QtGui, QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        object_dock = _entity_dock(widget, qapp, "object")
        object_id = object_dock.table.item(0, 1).text()
        object_data = _select_row(object_dock, qapp, object_id)
        assert object_data["original_filename"] == "sample_curve.csv"
        assert object_data["storage_path"]

        widget._copy_selected_object_entity_uuid()
        assert QtWidgets.QApplication.clipboard().text() == object_id

        opened_urls = []
        with mock.patch.object(
            QtGui.QDesktopServices,
            "openUrl",
            side_effect=lambda url: opened_urls.append(url) or True,
        ):
            widget._reveal_selected_object_entity()
        assert opened_urls
        assert opened_urls[0].isLocalFile()
        assert opened_urls[0].toLocalFile().endswith(object_data["storage_path"])

        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            widget._delete_selected_object_entity()
        qapp.processEvents()

        assert seeded_admin_db.get_object_info(object_id) is None
        assert object_dock.table.rowCount() == 0
        assert "Object delete result" in widget.status_label.text()


def test_branch_entity_dock_creates_updates_and_deletes_branch(seeded_admin_db, qapp):
    from qtpy import QtCore, QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        dock = _entity_dock(widget, qapp, "branch")
        original_count = dock.table.rowCount()

        dock._on_new()
        qapp.processEvents()

        created_id = dock.selected_row_id()
        assert created_id
        created = seeded_admin_db.get_branch(created_id)
        assert created is not None
        assert created["name"].startswith("untitled_branch_")
        assert dock.table.rowCount() == original_count + 1

        dock.form._model.description = "created and updated from GUI"
        qapp.processEvents()
        assert (
            seeded_admin_db.get_branch(created_id)["description"]
            == "created and updated from GUI"
        )

        checkbox = dock.table.item(dock.table.currentRow(), 0)
        checkbox.setCheckState(QtCore.Qt.Checked)
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            dock._on_delete()
        qapp.processEvents()

    assert seeded_admin_db.get_branch(created_id) is None


def test_data_product_and_analysis_entity_docks_browse_seeded_data(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        raw_dock = _entity_dock(widget, qapp, "raw_data")
        raw_data = _select_row(raw_dock, qapp, "raw_gui")
        assert raw_data["experiment_id"] == "exp_gui"
        assert raw_data["data_type"] == "tcspc"

        processing_dock = _entity_dock(widget, qapp, "processing_run")
        processing_data = _select_row(processing_dock, qapp, "proc_gui")
        assert processing_data["experiment_id"] == "exp_gui"
        assert processing_data["type"] == "filtering"

        product_dock = _entity_dock(widget, qapp, "processed_product")
        product_data = _select_row(product_dock, qapp, "prod_gui")
        assert product_data["processing_id"] == "proc_gui"
        assert product_data["product_type"] == "processed_data"

        analysis_dock = _entity_dock(widget, qapp, "analysis")
        analysis_data = _select_row(analysis_dock, qapp, "analysis_gui")
        assert analysis_data["experiment_id"] == "exp_gui"
        assert analysis_data["model_name"] == "GUI Fit"


def test_data_product_and_analysis_entity_dock_actions_use_seeded_data(
    seeded_admin_db,
    qapp,
):
    from qtpy import QtGui, QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        raw_dock = _entity_dock(widget, qapp, "raw_data")
        raw_buttons = {button.text() for button in raw_dock.findChildren(QtWidgets.QToolButton)}
        assert {"📋 Copy ID", "📂 Reveal", "🌱 Use as provenance seed"}.issubset(raw_buttons)
        raw_data = _select_row(raw_dock, qapp, "raw_gui")
        assert raw_data["location"].endswith("raw_gui.ptu")

        widget._copy_selected_raw_data_id()
        assert QtWidgets.QApplication.clipboard().text() == "raw_gui"

        opened_urls = []
        with mock.patch.object(
            QtGui.QDesktopServices,
            "openUrl",
            side_effect=lambda url: opened_urls.append(url) or True,
        ):
            widget._reveal_selected_raw_data()
        assert opened_urls
        assert opened_urls[0].isLocalFile()
        assert opened_urls[0].toLocalFile().endswith("raw_gui.ptu")

        widget._seed_selected_raw_data()
        assert widget.current_provenance_seed_type == "raw_data"
        assert widget.current_provenance_seed_id == "raw_gui"
        assert widget.prov_seed_id_edit.text() == "raw_gui"

        product_dock = _entity_dock(widget, qapp, "processed_product")
        product_buttons = {button.text() for button in product_dock.findChildren(QtWidgets.QToolButton)}
        assert {"📋 Copy ID", "📂 Reveal", "🌱 Use as provenance seed"}.issubset(product_buttons)
        product_data = _select_row(product_dock, qapp, "prod_gui")
        assert product_data["location"].endswith("prod_gui.json")

        widget._copy_selected_processed_product_id()
        assert QtWidgets.QApplication.clipboard().text() == "prod_gui"

        opened_urls.clear()
        with mock.patch.object(
            QtGui.QDesktopServices,
            "openUrl",
            side_effect=lambda url: opened_urls.append(url) or True,
        ):
            widget._reveal_selected_processed_product()
        assert opened_urls
        assert opened_urls[0].isLocalFile()
        assert opened_urls[0].toLocalFile().endswith("prod_gui.json")

        widget._seed_selected_processed_product()
        assert widget.current_provenance_seed_type == "processed_data"
        assert widget.current_provenance_seed_id == "prod_gui"
        assert widget.prov_seed_id_edit.text() == "prod_gui"

        analysis_dock = _entity_dock(widget, qapp, "analysis")
        analysis_buttons = {button.text() for button in analysis_dock.findChildren(QtWidgets.QToolButton)}
        assert {"📋 Copy ID", "🔎 Details", "🌱 Use as provenance seed"}.issubset(analysis_buttons)
        _select_row(analysis_dock, qapp, "analysis_gui")

        widget._copy_selected_analysis_id()
        assert QtWidgets.QApplication.clipboard().text() == "analysis_gui"

        widget._show_selected_analysis_details()
        qapp.processEvents()
        dialog = widget._analysis_drilldown_dialog
        parameter_table = dialog.findChild(QtWidgets.QTableWidget, "analysis_parameters_table")
        input_table = dialog.findChild(QtWidgets.QTableWidget, "analysis_input_products_table")
        output_table = dialog.findChild(QtWidgets.QTableWidget, "analysis_output_products_table")
        assert parameter_table.rowCount() == 1
        assert parameter_table.item(0, 0).text() == "tau"
        assert parameter_table.item(0, 1).text() == "3.8"
        assert input_table.rowCount() == 0
        assert output_table.rowCount() == 1
        assert output_table.item(0, 0).text() == "fit_result_gui"
        assert output_table.item(0, 3).text() == "valid"

        widget._seed_selected_analysis()
        assert widget.current_provenance_seed_type == "analysis_run"
        assert widget.current_provenance_seed_id == "analysis_gui"
        assert widget.prov_seed_id_edit.text() == "analysis_gui"


def test_raw_and_processed_entity_dock_validate_and_delete_seeded_artifacts(
    seeded_admin_db,
    qapp,
):
    from qtpy import QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        raw_dock = _entity_dock(widget, qapp, "raw_data")
        raw_buttons = {button.text() for button in raw_dock.findChildren(QtWidgets.QToolButton)}
        assert {"✓ Validate", "🗑 Delete"}.issubset(raw_buttons)
        _select_row(raw_dock, qapp, "raw_gui")

        with mock.patch.object(
            QtWidgets.QInputDialog,
            "getItem",
            return_value=("warning", True),
        ):
            widget._validate_selected_raw_data()
        qapp.processEvents()

        assert seeded_admin_db.get_artifact("raw_gui")["validation_status"] == "warning"
        assert "validation set to warning" in widget.status_label.text()

        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            widget._delete_selected_raw_data()
        qapp.processEvents()

        assert seeded_admin_db.get_artifact("raw_gui")["deleted_at"] is not None
        assert raw_dock.table.rowCount() == 0
        raw_link = seeded_admin_db.conn.execute(
            """SELECT deleted_at FROM mfdb_operation_artifact
               WHERE operation_id = 'proc_gui' AND artifact_id = 'raw_gui'"""
        ).fetchone()
        assert raw_link["deleted_at"] is not None

        product_dock = _entity_dock(widget, qapp, "processed_product")
        product_buttons = {button.text() for button in product_dock.findChildren(QtWidgets.QToolButton)}
        assert {"✓ Validate", "🗑 Delete"}.issubset(product_buttons)
        _select_row(product_dock, qapp, "prod_gui")

        with mock.patch.object(
            QtWidgets.QInputDialog,
            "getItem",
            return_value=("invalid", True),
        ):
            widget._validate_selected_processed_product()
        qapp.processEvents()

        assert seeded_admin_db.get_artifact("prod_gui")["validation_status"] == "invalid"
        assert "validation set to invalid" in widget.status_label.text()

        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            widget._delete_selected_processed_product()
        qapp.processEvents()

        assert seeded_admin_db.get_artifact("prod_gui")["deleted_at"] is not None
        visible_product_ids = {
            product_dock.table.item(row, 1).text()
            for row in range(product_dock.table.rowCount())
        }
        assert "prod_gui" not in visible_product_ids
        assert "fit_result_gui" in visible_product_ids
        product_link = seeded_admin_db.conn.execute(
            """SELECT deleted_at FROM mfdb_operation_artifact
               WHERE operation_id = 'proc_gui' AND artifact_id = 'prod_gui'"""
        ).fetchone()
        assert product_link["deleted_at"] is not None


def test_measurements_panel_refreshes_and_filters_seeded_data(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        _select_panel(widget, qapp, "Measurements")
        widget._refresh_measurements()
        qapp.processEvents()

        rows = {
            (
                widget.measurements_table.item(row, 0).text(),
                widget.measurements_table.item(row, 1).text(),
            )
            for row in range(widget.measurements_table.rowCount())
        }
        assert ("Raw", "raw_gui") in rows
        assert ("Processing", "proc_gui") in rows
        assert ("Processed", "prod_gui") in rows
        assert ("Processed", "fit_result_gui") in rows
        assert widget.status_label.text() == "Measurements: 4 rows"

        widget.meas_kind_combo.setCurrentText("Raw data")
        widget._refresh_measurements()
        qapp.processEvents()
        assert widget.measurements_table.rowCount() == 1
        assert widget.measurements_table.item(0, 0).text() == "Raw"
        assert widget.measurements_table.item(0, 1).text() == "raw_gui"

        widget.meas_kind_combo.setCurrentText("All")
        widget.meas_search_edit.setText("prod_gui")
        widget._refresh_measurements()
        qapp.processEvents()
        assert widget.measurements_table.rowCount() == 1
        assert widget.measurements_table.item(0, 0).text() == "Processed"
        assert widget.measurements_table.item(0, 1).text() == "prod_gui"


def test_provenance_graph_loads_seeded_processing_graph(seeded_admin_db, qapp):
    with _widget_for_db(seeded_admin_db) as widget:
        widget._set_provenance_seed("processed_data", "prod_gui")
        qapp.processEvents()

        widget.load_provenance_full_graph()
        qapp.processEvents()

        assert widget.prov_seed_type_combo.currentText() == "processed_data"
        assert widget.prov_seed_id_edit.text() == "prod_gui"
        assert widget.prov_edge_table.rowCount() == 2

        edge_rows = {
            (
                widget.prov_edge_table.item(row, 1).text(),
                widget.prov_edge_table.item(row, 2).text(),
                widget.prov_edge_table.item(row, 3).text(),
                widget.prov_edge_table.item(row, 4).text(),
                widget.prov_edge_table.item(row, 5).text(),
            )
            for row in range(widget.prov_edge_table.rowCount())
        }
        assert ("artifact", "raw_gui", "input_to", "operation", "proc_gui") in edge_rows
        assert ("operation", "proc_gui", "produced", "artifact", "prod_gui") in edge_rows

        graph = widget.prov_node_editor.graph_dict()
        assert {node["id"] for node in graph["nodes"]} == {
            "artifact:prod_gui",
            "artifact:raw_gui",
            "operation:proc_gui",
        }
        assert len(graph["edges"]) == 2

        widget.prov_edge_table.selectRow(0)
        widget.on_prov_edge_selected()
        qapp.processEvents()
        details = widget.prov_details_text.toPlainText()
        assert "relationship_type" in details
        assert "proc_gui" in details


def test_import_export_panel_validates_previews_and_exports_seeded_sample(
    seeded_admin_db,
    qapp,
    tmp_path,
):
    from qtpy import QtWidgets

    with _widget_for_db(seeded_admin_db) as widget:
        _select_panel(widget, qapp, "Import / Export")
        widget.sample_id_edit.setText("sample_gui")
        qapp.processEvents()

        widget.validate_selected_sample_export()
        qapp.processEvents()
        validation_text = widget.preview_edit.toPlainText()
        assert '"valid"' in validation_text
        assert '"valid": false' in validation_text
        assert widget.status_label.text().startswith("Export validation for sample_gui:")

        widget.preview_cif()
        qapp.processEvents()
        assert widget.status_label.text() == "CIF preview for sample_gui"
        assert widget.preview_edit.toPlainText() is not None

        sample_path = tmp_path / "sample_gui.cif"
        table_path = tmp_path / "samples.csv"
        with mock.patch.object(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=(str(sample_path), "CIF files (*.cif *.mmcif)"),
        ), mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            widget.export_selected_sample()
        qapp.processEvents()
        assert sample_path.exists()
        assert widget.status_label.text() == f"Exported {sample_path}"

        with mock.patch.object(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=(str(table_path), "CSV/Excel files (*.csv *.tsv *.xlsx)"),
        ):
            widget.export_table()
        qapp.processEvents()
        assert table_path.exists()
        assert widget.status_label.text() == f"Exported {table_path}"
        assert "sample_gui" in table_path.read_text(encoding="utf-8")

        import_path = tmp_path / "gui_import_sample.cif"
        import_path.write_text("data_gui_import_sample\n#\n", encoding="utf-8")
        widget.file_edit.setText(str(import_path))
        widget.import_file()
        qapp.processEvents()
        assert seeded_admin_db.get_sample("gui_import_sample") is not None
        assert "gui_import_sample" in widget.preview_edit.toPlainText()
