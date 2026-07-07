"""MFDB registration for Micro-time Shifter results."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chisurf.core.mfdb.provenance.result_registry import (
    register_raw_measurement,
    register_result,
)

from .contract import CONTRACT_VERSION
from .models import ShiftRequest, ShiftResult

if TYPE_CHECKING:
    from chisurf.core.mfdb.security.base import MFDBClientBase

logger = logging.getLogger(__name__)


@dataclass
class ShiftRegistrationResult:
    """MFDB artifact IDs created for one micro-time shift run.

    Attributes
    ----------
    input_artifacts : dict
        Raw input artifacts keyed by input file path.
    output_artifacts : dict
        Shifted output artifacts keyed by input file path.
    warnings : list
        Non-fatal registration warnings.

    """

    input_artifacts: dict[str, str] = field(default_factory=dict)
    output_artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _file_md5(path: str) -> str:
    """Return the MD5 content hash for a file.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        Hex-encoded MD5 digest.

    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_mfdb_connection() -> "MFDBClientBase | None":
    """Return the active/global MFDB connection, or ``None``.

    Connection acquisition is api-layer (not view) logic (PRD-23); the GUI tool
    reads the ambient connection through this helper instead of importing the
    result registry directly.
    """
    try:
        from chisurf.core.mfdb.provenance.result_registry import _get_global_db

        return _get_global_db()
    except Exception:
        return None


class MicrotimeShiftMFDBPipeline:
    """Register Micro-time Shifter inputs and outputs in MFDB."""

    def __init__(self, db: MFDBClientBase | None = None):
        """Create a shift registration pipeline.

        Parameters
        ----------
        db : MFDBClientBase, optional
            Explicit MFDB connection. If omitted, the result registry
            resolves the active/global database.

        """
        if db is None:
            db = active_mfdb_connection()
        self.db = db

    def _find_raw_artifact_by_md5(self, md5: str) -> str:
        """Look up an existing raw artifact by content hash.

        Parameters
        ----------
        md5 : str
            MD5 content hash.

        Returns
        -------
        str
            Artifact ID, or empty string if not found.

        """
        if self.db is None:
            return ""
        try:
            return self.db.find_raw_artifact_by_md5(md5)
        except Exception:
            return ""

    def _lookup_or_register_raw(
        self,
        path: str,
        request: ShiftRequest,
    ) -> tuple[str, bool]:
        """Look up or register a raw input artifact.

        Parameters
        ----------
        path : str
            Input file path.
        request : ShiftRequest
            Shift request with MFDB context.

        Returns
        -------
        tuple of (str, bool)
            Artifact ID and whether it was newly registered.

        """
        md5 = _file_md5(path)
        existing = self._find_raw_artifact_by_md5(md5)
        if existing:
            # Content already registered (dedup): record this user as a
            # co-owner so the dataset appears under their "Mine" scope too.
            try:
                from chisurf.core.mfdb.provenance.result_registry import _resolve_active_user_id
                if self.db is not None:
                    self.db.add_artifact_owner(existing, _resolve_active_user_id())
            except Exception:
                pass
            return existing, False

        if not request.mfdb.register_missing_inputs:
            return "", False

        artifact_id = register_raw_measurement(
            file_path=path,
            sample_id=request.mfdb.sample_id,
            metadata={
                "plugin": "microtime_shifter",
                "role": "raw_tttr",
                "filetype": request.filetype,
            },
            setup_id=request.mfdb.setup_id,
            setup_version=request.mfdb.setup_version,
            db=self.db,
        )
        return artifact_id, bool(artifact_id)

    def register_run(
        self,
        request: ShiftRequest,
        result: ShiftResult,
    ) -> ShiftRegistrationResult:
        """Register shift inputs and outputs in MFDB.

        Parameters
        ----------
        request : ShiftRequest
            Original shift request.
        result : ShiftResult
            Successful shift result.

        Returns
        -------
        ShiftRegistrationResult
            Created artifact IDs and warnings.

        """
        registration = ShiftRegistrationResult()
        if not request.mfdb.enabled:
            return registration

        for input_file in request.files:
            norm_path = str(Path(input_file).resolve())
            raw_id, is_new = self._lookup_or_register_raw(norm_path, request)
            if not raw_id:
                registration.warnings.append(
                    f"MFDB registration skipped for input {norm_path}."
                )
                continue
            registration.input_artifacts[norm_path] = raw_id

            shifted_path = result.output_paths_by_file.get(norm_path)
            if not shifted_path:
                continue

            applied = result.applied_shifts_by_file.get(norm_path, {})
            # Operation parameters per the .dic schema for operation_type
            # "microtime_shift": a scalar global_shift and the repeatable,
            # role-indexed shift (one entry per detector channel, role = channel).
            # This replaces both the flat shift_ch<N> names and the bespoke
            # mfdb_microtime_shift table (retired) with role-indexed mfdb_parameter
            # rows recorded by register_result/_record_parameters.
            param_dict: dict[str, Any] = {
                "global_shift": applied.get("global_shift", 0),
                "shift": [
                    {"value": sv, "role": str(ch)}
                    for ch, sv in applied.get("channel_shifts", {}).items()
                ],
            }

            try:
                artifact_id = register_result(
                    kind="processed_data",
                    data=shifted_path,
                    sample_id=request.mfdb.sample_id,
                    parent_artifact_id=raw_id,
                    operation_type="microtime_shift",
                    parameters=param_dict,
                    metadata={
                        "plugin": "microtime_shifter",
                        "contract_version": CONTRACT_VERSION,
                        "input_file": norm_path,
                        "global_shift": applied.get("global_shift", 0),
                        "channel_shifts": {
                            str(k): v
                            for k, v in applied.get("channel_shifts", {}).items()
                        },
                    },
                    data_format=Path(shifted_path).suffix.lstrip(".") or "tttr",
                    setup_id=request.mfdb.setup_id,
                    setup_version=request.mfdb.setup_version,
                    db=self.db,
                )
                if artifact_id:
                    registration.output_artifacts[norm_path] = artifact_id
                else:
                    registration.warnings.append(
                        f"MFDB did not register shifted output for {norm_path}."
                    )
            except Exception as exc:
                registration.warnings.append(
                    f"MFDB registration error for {norm_path}: {exc}"
                )

        return registration
