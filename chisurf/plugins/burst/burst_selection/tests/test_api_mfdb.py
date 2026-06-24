"""PRD-23: the raw-input MFDB helpers live in the api layer (Qt-free).

These helpers used to live in ``gui/tool.py`` (with ``register_result``/``MFDatabase``
imported into the view). They were moved to ``api/mfdb.py`` so the GUI calls the api
only; this module exercises them headlessly (no Qt), which also asserts the api/cli
layer imports without a Qt binding.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.plugins.burst.burst_selection.api.mfdb import (
    file_md5,
    raw_artifact_id_for_path,
    raw_file_data_format,
    register_raw_input_for_sample,
    sample_id_for_raw_path,
)


@pytest.fixture
def db(tmp_path) -> MFDatabase:
    database = MFDatabase(os.path.join(tmp_path, "mfdb.db"))
    yield database
    database.close()


def _raw(tmp_path: Path, payload: bytes = b"\x00\x01\x02\x03") -> Path:
    path = tmp_path / "m.ptu"
    path.write_bytes(payload)
    return path


def test_file_md5_matches_hashlib(tmp_path):
    path = _raw(tmp_path, b"abc")
    assert file_md5(path) == hashlib.md5(b"abc").hexdigest()


def test_raw_file_data_format_vocabulary():
    assert raw_file_data_format(Path("x.h5")) == "hdf5"
    assert raw_file_data_format(Path("x.ht3")) == "tttr"
    assert raw_file_data_format(Path("x.ptu")) == "ptu"
    assert raw_file_data_format(Path("x")) == "tttr"


def test_register_binds_artifact_to_sample_and_is_discoverable(db, tmp_path):
    raw = _raw(tmp_path)
    db.add_sample("s1", description="S", num_of_probes=1)
    artifact_id = register_raw_input_for_sample(
        db=db, path=raw, sample_id="s1", filetype="ptu", selected_setup=None
    )
    assert artifact_id
    # content-addressed lookups resolve the same artifact / sample
    assert raw_artifact_id_for_path(db, raw) == artifact_id
    assert sample_id_for_raw_path(db, raw) == "s1"


def test_register_missing_sample_returns_empty(db, tmp_path):
    raw = _raw(tmp_path)
    assert (
        register_raw_input_for_sample(
            db=db, path=raw, sample_id="nope", filetype=None, selected_setup=None
        )
        == ""
    )
