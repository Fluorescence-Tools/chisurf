"""Experiment readers and their optional GUI controllers.

This module defines abstract base classes that connect low-level I/O
(:class:`ExperimentReader`) with higher-level controllers
(:class:`ExperimentReaderController`). Concrete readers are responsible
for turning files into :class:`chisurf.core.data.ExperimentalData` objects.

When ``record_provenance`` is enabled and a database connection is available,
:meth:`ExperimentReader.get_data` automatically registers source files and
derived data in the object store and records provenance in MFDB.
"""
from __future__ import annotations

import abc
import hashlib
import logging
import pathlib
import uuid

import chisurf.core.base
import chisurf.core.curve
import chisurf.core.data
from chisurf import typing

logger = logging.getLogger(__name__)


class ExperimentReader(chisurf.core.base.Base):
    """Abstract base class for loading experimental data.

    Subclasses implement :meth:`read` (file I/O) and
    :meth:`autofitrange`. The helper :meth:`get_data` wraps a single
    dataset into an :class:`chisurf.core.data.ExperimentDataGroup` and
    attaches experiment and setup references.

    When ``record_provenance`` is True and ``db`` is set, ``get_data()``
    automatically registers source files and derived data in the object
    store and records provenance in MFDB.

    Class Attributes
    ----------------
    operation_type : str
        Operation type for provenance recording.
    artifact_kind_source : str
        Artifact kind for source files.
    artifact_kind_derived : str
        Artifact kind for derived data.
    derived_data_format : str
        Format string for derived data (e.g. ``'json'``).
    derived_mime_type : str
        MIME type for derived data.

    Examples
    --------
    Create a tiny reader that returns a synthetic curve and expose it as
    a one-element :class:`chisurf.core.data.ExperimentDataGroup`:

    >>> import numpy as np
    >>> class _DummyReader(ExperimentReader):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         self.experiment = None
    ...     def autofitrange(self, data, **kwargs):
    ...         return 0, len(data)
    ...     def read(self, *args, **kwargs):
    ...         return chisurf.core.data.DataCurve(x=np.array([0.0, 1.0]),
    ...                                      y=np.array([1.0, 2.0]))
    >>> r = _DummyReader(name="dummy")
    >>> group = r.get_data()
    >>> len(group)
    1
    """

    controller: ExperimentReaderController = None

    operation_type: str = "measurement_import"
    artifact_kind_source: str = "raw_data"
    artifact_kind_derived: str = "processed_data"
    derived_data_format: str = "json"
    derived_mime_type: str = "application/json"

    def __init__(
            self,
            *args,
            controller: ExperimentReaderController = None,
            db=None,
            object_store=None,
            record_provenance: bool = True,
            **kwargs
    ):
        """Initialize the experiment reader.

        Parameters
        ----------
        controller : ExperimentReaderController, optional
            Associated controller instance.
        db : MFDatabase, optional
            Database connection for provenance recording.
        object_store : ObjectStore, optional
            Object store for content-addressed storage.
        record_provenance : bool
            Whether to record provenance when ``get_data()`` is called.
        """
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.db = db
        self.object_store = object_store
        self.record_provenance = record_provenance
        self.sample_id: str | None = None
        self._last_operation_id: str | None = None
        self._source_md5s: dict[str, str] = {}
        self._source_object_uuids: list[str] = []
        self._pending_sample_files: list[tuple[str, str, str]] = []
        self._derived_object_uuids: list[str] = []

    @abc.abstractmethod
    def autofitrange(self, data: chisurf.core.base.Data, **kwargs) -> typing.Tuple[int, int]:
        """Determine the default fit range for a dataset.

        Parameters
        ----------
        data : chisurf.core.base.Data
            The experimental data to estimate a fit range for.

        Returns
        -------
        tuple of int
            ``(start, stop)`` indices defining the fit interval.
        """
        if isinstance(data, chisurf.core.data.DataCurve):
            return 0, len(data)
        else:
            return 0, 0

    def __getstate__(self):
        """Serialize the reader state, skipping Qt widgets.

        Returns
        -------
        dict
            Pickle-friendly state dictionary.
        """
        state = super().__getstate__()
        state.update(
            chisurf.core.base.to_elementary(self.__dict__.copy(), skip_qt_widgets=True)
        )
        return state

    @abc.abstractmethod
    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.core.base.Data:
        """Read experimental data from a file.

        Parameters
        ----------
        filename : str, optional
            Path to the data file.

        Returns
        -------
        chisurf.core.base.Data
            The loaded experimental data.
        """
        pass

    def get_data(self, **kwargs) -> chisurf.core.data.ExperimentDataGroup:
        """Read data and wrap it in an :class:`ExperimentDataGroup`.

        Attaches the experiment, setup, and data_reader references
        to each dataset in the group. When ``record_provenance`` is
        enabled, also registers source files and derived data in the
        object store and records provenance in MFDB.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`read`.

        Returns
        -------
        chisurf.core.data.ExperimentDataGroup
            Group containing the loaded data with metadata attached.
        """
        filename = kwargs.get("filename")
        source_uuids = self._register_sources(filename)

        data = self.read(**kwargs)

        derived_uuids = self._register_derived(data)
        op_id = self._record_read_operation(source_uuids, derived_uuids)

        if isinstance(data, chisurf.core.data.ExperimentalData):
            data = chisurf.core.data.ExperimentDataGroup([data])
        if isinstance(data, chisurf.core.data.ExperimentDataGroup):
            for d in data:
                d.experiment = self.experiment
                d.setup = self
                try:
                    d.data_reader = self
                except Exception:
                    pass
            try:
                if not hasattr(data, "data_reader"):
                    data.data_reader = self
            except Exception:
                pass

        self._stamp_data(data, source_uuids, derived_uuids, op_id)
        return data

    def _prompt_for_sample_batch(self, pending):
        """Prompt the user to assign a sample for each newly loaded file."""
        for filename, content_md5, object_uuid in pending:
            self._prompt_for_sample(filename, content_md5, object_uuid)

    def _prompt_for_sample(self, filename, content_md5, object_uuid):
        """Prompt the user to assign a sample if the file is new to MFDB."""
        try:
            from chisurf.gui.widgets.experiments.sample_selector_widget import show_sample_lookup_dialog
            show_sample_lookup_dialog(
                self.controller,
                str(filename),
                content_md5,
                object_uuid,
            )
        except Exception:
            pass

    def _register_sources(self, filename) -> list[str]:
        """Register source file(s) in the object store.

        Parameters
        ----------
        filename : str or list of str or None
            Path(s) to source file(s).

        Returns
        -------
        list of str
            Object UUIDs of registered source files.
        """
        if not self.record_provenance or self.db is None:
            return []
        paths = self._normalize_filenames(filename)
        uuids = []
        pending = []
        for p in paths:
            if not p.is_file():
                continue
            try:
                result = self.db.put_object(path=str(p), filename=str(p))
                obj_uuid = result["object_uuid"]
                content_md5 = result.get("content_md5")
                self._source_md5s[str(p)] = content_md5 or ""
                uuids.append(obj_uuid)
                sample_id = None
                if content_md5:
                    sample_id = self.db.lookup_sample_by_md5(content_md5)
                if sample_id:
                    self.sample_id = sample_id
                elif self.sample_id:
                    sample_id = self.sample_id
                if sample_id:
                    self.db.set_object_sample_id(obj_uuid, sample_id)
                else:
                    pending.append((str(p), content_md5 or "", obj_uuid))
                artifact_id = f"src_{obj_uuid[:12]}"
                self.db.register_artifact(
                    artifact_id=artifact_id,
                    artifact_kind=self.artifact_kind_source,
                    storage_mode="local_file",
                    file_path=str(p),
                    object_uuid=obj_uuid,
                    data_format=p.suffix.lstrip(".") or "unknown",
                    size_bytes=result["size_bytes"],
                )
            except Exception as e:
                logger.warning("Failed to register source %s: %s", p, e)
        self._source_object_uuids = uuids
        self._pending_sample_files = pending
        if pending and self.controller is not None:
            self._prompt_for_sample_batch(pending)
        return uuids

    def _register_derived(self, data) -> list[str]:
        """Serialize and register derived data in the object store.

        Each DataCurve gets its own object UUID.

        Parameters
        ----------
        data : Data
            The computed data to register.

        Returns
        -------
        list of str
            Object UUIDs of registered derived objects.
        """
        if not self.record_provenance or self.db is None:
            return []
        from chisurf.core.experiments.core.serialize import data_to_json

        uuids = []
        curves = self._extract_curves(data)
        for curve in curves:
            try:
                blob = data_to_json(
                    curve,
                    data_type=self.artifact_kind_derived,
                    created_by=type(self).__name__,
                    source_object_uuids=self._source_object_uuids,
                    reader_settings=self._reader_settings_dict(),
                )
                result = self.db.put_object(
                    data=blob,
                    filename=f"{getattr(curve, 'name', 'derived')}.json",
                    mime_type=self.derived_mime_type,
                )
                obj_uuid = result["object_uuid"]
                uuids.append(obj_uuid)
                artifact_id = f"der_{obj_uuid[:12]}"
                self.db.register_artifact(
                    artifact_id=artifact_id,
                    artifact_kind=self.artifact_kind_derived,
                    storage_mode="embedded_json",
                    object_uuid=obj_uuid,
                    data_format=self.derived_data_format,
                    size_bytes=result["size_bytes"],
                )
            except Exception as e:
                logger.warning("Failed to register derived data: %s", e)
        self._derived_object_uuids = uuids
        return uuids

    def _record_read_operation(
        self, source_uuids: list[str], derived_uuids: list[str]
    ) -> str | None:
        """Record an MFDB operation linking source and derived artifacts.

        Parameters
        ----------
        source_uuids : list of str
            Object UUIDs of source files.
        derived_uuids : list of str
            Object UUIDs of derived data.

        Returns
        -------
        str or None
            The operation ID, or None if provenance recording is disabled.
        """
        if not self.record_provenance or self.db is None:
            return None
        if not source_uuids and not derived_uuids:
            return None

        op_id = f"op_{uuid.uuid4().hex[:12]}"
        try:
            input_artifacts = [
                {"artifact_id": f"src_{u[:12]}", "role": "source_file"}
                for u in source_uuids
            ]
            output_artifacts = [
                {"artifact_id": f"der_{u[:12]}", "role": "derived_data"}
                for u in derived_uuids
            ]
            self.db.record_operation_with_artifacts(
                operation_id=op_id,
                operation_type=self.operation_type,
                status="success",
                input_artifacts=input_artifacts,
                output_artifacts=output_artifacts,
                settings=self._reader_settings_dict(),
                software_module=type(self).__name__,
            )
        except Exception as e:
            logger.warning("Failed to record operation: %s", e)
            return None
        self._last_operation_id = op_id
        return op_id

    def _stamp_data(
        self,
        data,
        source_uuids: list[str],
        derived_uuids: list[str],
        op_id: str | None,
    ) -> None:
        """Attach provenance UUIDs to returned data objects.

        Parameters
        ----------
        data : Data
            The data to stamp.
        source_uuids : list of str
            Source object UUIDs.
        derived_uuids : list of str
            Derived object UUIDs.
        op_id : str or None
            Operation ID.
        """
        if not source_uuids and not derived_uuids and op_id is None:
            return
        provenance = {
            "source_object_uuids": source_uuids,
            "derived_object_uuids": derived_uuids,
            "operation_id": op_id,
        }
        curves = self._extract_curves(data)
        for i, curve in enumerate(curves):
            if not hasattr(curve, "meta_data") or curve.meta_data is None:
                curve.meta_data = {}
            curve.meta_data["mfdb"] = {
                **provenance,
                "derived_object_uuid": derived_uuids[i] if i < len(derived_uuids) else None,
            }

    def _normalize_filenames(self, filename) -> list[pathlib.Path]:
        """Normalize filename argument to a list of Paths."""
        if filename is None:
            return []
        if isinstance(filename, (str, pathlib.Path)):
            return [pathlib.Path(str(filename))]
        if isinstance(filename, (list, tuple)):
            return [pathlib.Path(str(f)) for f in filename]
        return []

    def _extract_curves(self, data) -> list:
        """Extract individual DataCurves from data."""
        if isinstance(data, chisurf.core.data.DataCurve):
            return [data]
        if isinstance(data, (chisurf.core.data.DataGroup, chisurf.core.data.ExperimentDataGroup)):
            curves = []
            for item in data:
                if isinstance(item, chisurf.core.data.DataCurve):
                    curves.append(item)
                elif isinstance(item, chisurf.core.data.DataCurveGroup):
                    for sub in item:
                        if isinstance(sub, chisurf.core.data.DataCurve):
                            curves.append(sub)
            return curves
        return []

    def _reader_settings_dict(self) -> dict:
        """Serialize reader configuration for provenance recording."""
        state = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or key in ("db", "object_store", "controller", "experiment"):
                continue
            try:
                state[key] = chisurf.core.base.to_elementary(value, skip_qt_widgets=True)
            except Exception:
                state[key] = str(value)
        return state


class ExperimentReaderController(chisurf.core.base.Base):
    """Base class that couples a reader to higher-level UIs.

    The controller can manage file selection and other GUI-related
    aspects while delegating computational work to the underlying
    :class:`ExperimentReader` instance.
    """

    experiment_reader: ExperimentReader = None

    def __init__(self, experiment_reader: ExperimentReader = None, *args, **kwargs):
        """Initialize the reader controller.

        Parameters
        ----------
        experiment_reader : ExperimentReader, optional
            The reader instance the controller manages.
        """
        super().__init__(*args, **kwargs)
        self.experiment_reader = experiment_reader
        self._call_dict = dict()
        if isinstance(experiment_reader, ExperimentReader):
            experiment_reader.controller = self

    def __getattr__(self, item: str):
        """Delegate attribute access to the underlying experiment reader."""
        return getattr(self.experiment_reader, item)

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        """The filename of the currently selected data file."""
        pass

    @abc.abstractmethod
    def get_filename(self) -> pathlib.Path:
        """Return the path to the currently selected data file.

        Returns
        -------
        pathlib.Path
            Path object for the selected file.
        """
        pass
