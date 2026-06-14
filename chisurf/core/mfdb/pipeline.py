from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.graph import traverse_canonical_graph
from chisurf.plugins.burst.burst_selection.backend.services import analyze_files_handler


class BurstPipeline:
    """A high-level in-process database pipeline for burst selection.

    This class runs burst selection workflows and registers the inputs,
    operations, and output artifacts directly in the provided MFDB database.
    """

    def __init__(self, db: MFDatabase):
        if not isinstance(db, MFDatabase):
            raise TypeError("db must be an instance of MFDatabase")
        self.db = db
        self.run_suffix = uuid.uuid4().hex[:8]

        self.raw_artifact_id = f"raw_spc_{self.run_suffix}"
        self.burst_operation_id = f"burst_select_op_{self.run_suffix}"
        self.bur_artifact_id = f"processed_bur_{self.run_suffix}"

        self.bur_file_path: str | None = None

    def run(self, spc_file: str | Path, settings: dict[str, Any] | None = None, **kwargs: Any) -> str:
        """Register input raw file, run burst selection, and register output processed file.

        Parameters
        ----------
        spc_file : str or Path
            Path to the input SPC file.
        settings : dict, optional
            A dictionary of burst selection settings.
        **kwargs : Any
            Individual parameter overrides (e.g. min_photons=20, time_window=1e-3).
        """
        spc_path = Path(spc_file).resolve()
        if not spc_path.exists():
            raise FileNotFoundError(f"SPC file not found: {spc_path}")

        # Calculate input raw file checksum
        with open(spc_path, "rb") as f:
            raw_checksum = hashlib.sha256(f.read()).hexdigest()

        # Register raw data artifact directly in DB
        self.db.register_artifact(
            artifact_id=self.raw_artifact_id,
            artifact_type="raw_data",
            storage_mode="local",
            file_path=str(spc_path),
            size_bytes=spc_path.stat().st_size,
            checksum=raw_checksum,
            checksum_algorithm="sha256",
            validation_status="valid",
            metadata={"description": "smDNA SPC measurement"},
        )

        # Setup settings structure
        actual_settings = {
            "photon_filter": {
                "channels": [0, 1, 8, 9],
                "filter_active": False,
                "delta_macro_time_filter": {"dT_min": 0.0},
            },
            "burst_detection": {
                "min_photons": 20,
                "photon_window": 10,
                "time_window": 1e-3,
            },
        }

        if settings:
            for k, v in settings.items():
                if isinstance(v, dict) and k in actual_settings:
                    actual_settings[k].update(v)
                else:
                    actual_settings[k] = v

        for key, val in kwargs.items():
            if key in ["min_photons", "photon_window", "time_window"]:
                actual_settings["burst_detection"][key] = val
            elif key in ["channels", "filter_active"]:
                actual_settings["photon_filter"][key] = val
            elif key == "dT_min":
                actual_settings["photon_filter"]["delta_macro_time_filter"]["dT_min"] = val
            else:
                actual_settings[key] = val

        # Record pending operation
        self.db.record_operation(
            operation_id=self.burst_operation_id,
            operation_type="burst_selection",
            settings=actual_settings,
            status="pending",
            software_module="burst_selection",
        )

        # Link input raw data to operation
        self.db.record_operation_link(
            operation_id=self.burst_operation_id,
            artifact_id=self.raw_artifact_id,
            direction="input",
            role="raw_tttr",
            checksum_snapshot=raw_checksum,
        )

        # Execute burst selection handler directly
        temp_dir = tempfile.mkdtemp()
        analysis_res = analyze_files_handler(
            files=[str(spc_path)],
            output_dir=temp_dir,
            settings=actual_settings,
        )

        if not analysis_res.get("ok"):
            raise RuntimeError(f"Analysis failed: {analysis_res.get('error')}")

        res_payload = analysis_res["result"]
        self.bur_file_path = res_payload["output_paths"]["bur"]

        with open(self.bur_file_path, "rb") as f:
            bur_checksum = hashlib.sha256(f.read()).hexdigest()

        # Register output processed bur file
        self.db.register_artifact(
            artifact_id=self.bur_artifact_id,
            artifact_type="processed_data",
            storage_mode="local",
            file_path=str(self.bur_file_path),
            size_bytes=os.path.getsize(self.bur_file_path),
            checksum=bur_checksum,
            checksum_algorithm="sha256",
            validation_status="valid",
            metadata={
                "description": "Output burst table",
                "n_bursts": res_payload["metadata"].get("n_bursts"),
            },
        )

        # Update operation to success
        self.db.record_operation(
            operation_id=self.burst_operation_id,
            operation_type="burst_selection",
            status="success",
            settings=actual_settings,
        )

        # Link output bur file
        self.db.record_operation_link(
            operation_id=self.burst_operation_id,
            artifact_id=self.bur_artifact_id,
            direction="output",
            role="burst_table",
            checksum_snapshot=bur_checksum,
        )

        return self.bur_file_path

    def get_lineage(self) -> list[dict[str, Any]]:
        """Retrieve upstream provenance lineage of the processed burst results."""
        if not self.bur_artifact_id:
            raise RuntimeError("No burst results registered yet.")

        return traverse_canonical_graph(
            self.db.conn,
            start_node_type="artifact",
            start_node_id=self.bur_artifact_id,
            direction="upstream",
        )
