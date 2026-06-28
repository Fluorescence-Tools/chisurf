"""Entity registry — wiring-only declarations for the dictionary-driven GUI.

This is the **only** per-entity hardcoding in the mfdb-admin GUI. It contains
no field/column data — only wiring: which RPC namespace, which dictionary
category, and which group an entity belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntitySpec:
    """Wiring specification for one MFDB entity type.

    Attributes
    ----------
    key : str
        Internal identifier (e.g. ``"sample"``, ``"experiment"``).
    title : str
        Dock title displayed in the UI (e.g. ``"Samples"``).
    category : str
        Dictionary category name (e.g. ``"flr_sample"``).
    rpc : str
        RPC namespace — resolves to ``mfdb.{rpc}.list/get/save/delete``.
    id_field : str
        Name of the primary key field in live records.
    group : str
        Logical category group: ``"Samples & chemistry"`` |
        ``"Experiments & data"`` | ``"Provenance"`` | ``"Administration"``.
    list_key : str
        Response key for list calls (default: same as ``rpc``).
    item_key : str
        Response key for get/save (default: singular form).
    writable : bool
        Whether this entity supports create/save/delete. True by default.
    """
    key: str
    title: str
    category: str
    rpc: str
    id_field: str
    group: str
    list_key: str = ""
    item_key: str = ""
    writable: bool = True
    schema_type: str = ""  # Fallback to legacy SCHEMAS type if dictionary has no category

    def __post_init__(self) -> None:
        if not self.list_key:
            object.__setattr__(self, "list_key", self.rpc)
        if not self.item_key:
            object.__setattr__(self, "item_key", self.rpc.rstrip("s"))


# The canonical registry. This list is the single source of truth for
# which entities exist and how to reach them on the backend.
ENTITY_REGISTRY: list[EntitySpec] = [
    # ---- Samples & chemistry ----
    EntitySpec(
        key="sample",
        title="Samples",
        category="flr_sample",
        rpc="samples",
        id_field="sample_id",
        group="Samples & chemistry",
        schema_type="sample",   # bridge: flr_sample dic is sparse; legacy covers FK fields
    ),
    EntitySpec(
        key="condition",
        title="Sample Conditions",
        category="flr_sample_condition",
        rpc="sample_conditions",
        id_field="condition_id",
        group="Samples & chemistry",
        schema_type="condition",
    ),
    EntitySpec(
        key="entity",
        title="Entities",
        category="flr_entity_assembly",
        rpc="entities",
        id_field="entity_id",
        group="Samples & chemistry",
    ),
    EntitySpec(
        key="probe",
        title="Probes",
        category="flr_probe_list",
        rpc="probes",
        id_field="probe_id",
        group="Samples & chemistry",
        schema_type="probe",  # bridge: dic category doesn't map optical-property columns
    ),
    EntitySpec(
        key="position",
        title="Label Positions",
        category="flr_poly_probe_position",
        rpc="probes.positions",
        id_field="id",
        list_key="positions",
        group="Samples & chemistry",
    ),
    EntitySpec(
        key="fret_pair",
        title="FRET Pairs",
        category="flr_fret_forster_radius",
        rpc="fret_pairs",
        id_field="forster_radius_id",
        group="Samples & chemistry",
    ),
    # NOTE: "metadata" is NOT a generic entity. Sample metadata uses
    # a per-sample key-value pattern (flr_sample_key_value table) that
    # doesn't fit the generic EntityDock CRUD model. It gets a dedicated
    # MetadataDock (see metadata_dock.py) wired separately in tool.py.
    # ---- Experiments & data ----
    EntitySpec(
        key="experiment",
        title="Experiments",
        category="flr_experiment",
        rpc="experiments",
        id_field="experiment_id",
        group="Experiments & data",
        schema_type="experiment",  # bridge: dic sparse on sample_id/device/user FKs
    ),
    EntitySpec(
        key="experiment_type",
        title="Experiment Types",
        category="flr_experiment_type",
        rpc="experiment_types",
        id_field="type_id",
        group="Experiments & data",
        schema_type="experiment_type",
    ),
    EntitySpec(
        key="setup",
        title="Setups",
        category="flr_inst_setting",
        rpc="setups",
        id_field="setup_id",
        group="Experiments & data",
        schema_type="setup",
    ),
    EntitySpec(
        key="detector_channel",
        title="Detector Channels",
        category="mfdb_setup_detector_channel",
        rpc="setups.detector_channels",
        id_field="id",
        list_key="detector_channels",
        group="Experiments & data",
        writable=False,
        schema_type="mfdb_setup_detector_channel",
    ),
    EntitySpec(
        key="pie_window",
        title="PIE Windows",
        category="mfdb_setup_pie_window",
        rpc="setups.pie_windows",
        id_field="id",
        list_key="pie_windows",
        group="Experiments & data",
        writable=False,
        schema_type="mfdb_setup_pie_window",
    ),
    EntitySpec(
        key="fcs_pair",
        title="FCS Pairs",
        category="mfdb_setup_fcs_pair",
        rpc="setups.fcs_pairs",
        id_field="id",
        list_key="fcs_pairs",
        group="Experiments & data",
        writable=False,
        schema_type="mfdb_setup_fcs_pair",
    ),
    EntitySpec(
        key="device",
        title="Devices",
        category="flr_instrument",
        rpc="devices",
        id_field="device_id",
        group="Experiments & data",
        schema_type="device",
    ),
    EntitySpec(
        key="raw_data",
        title="Raw Data",
        category="flr_raw_data",
        rpc="raw_data",
        id_field="raw_data_id",
        group="Experiments & data",
        writable=False,
        schema_type="raw_data",
    ),
    EntitySpec(
        key="processing_run",
        title="Processing Runs",
        category="flr_processing",
        rpc="processing",
        id_field="processing_id",
        group="Experiments & data",
        writable=False,
        schema_type="processing_run",
    ),
    EntitySpec(
        key="processed_product",
        title="Processed Products",
        category="flr_processed_data",
        rpc="processed_data",
        id_field="product_id",
        group="Experiments & data",
        writable=False,
        schema_type="processed_product",
    ),
    EntitySpec(
        key="analysis",
        title="Analyses",
        category="flr_analysis",
        rpc="analysis",
        id_field="analysis_id",
        group="Experiments & data",
        writable=False,
        schema_type="analysis",
    ),
    EntitySpec(
        key="object",
        title="Objects",
        category="mfdb_object",
        rpc="objects",
        id_field="object_uuid",
        group="Experiments & data",
        writable=False,
        schema_type="object",
    ),
    # ---- Provenance ----
    EntitySpec(
        key="project",
        title="Projects",
        category="flr_project",
        rpc="projects",
        id_field="project_id",
        group="Provenance",
        writable=False,
        schema_type="project",
    ),
    EntitySpec(
        key="branch",
        title="Branches",
        category="flr_branch",
        rpc="branches",
        id_field="branch_uuid",
        group="Provenance",
        schema_type="branch",
    ),
    # ---- Administration ----
    EntitySpec(
        key="user",
        title="Users",
        category="mfdb_user",
        rpc="users",
        id_field="user_id",
        group="Administration",
        schema_type="user",
    ),
]


def build_registry_dict() -> dict[str, dict[str, Any]]:
    """Build a dict form of the registry for FK resolution."""
    return {spec.key: {
        "key": spec.key,
        "title": spec.title,
        "category": spec.category,
        "rpc": spec.rpc,
        "id_field": spec.id_field,
        "group": spec.group,
        "list_key": spec.list_key,
        "item_key": spec.item_key,
        "writable": spec.writable,
        "schema_type": spec.schema_type,
    } for spec in ENTITY_REGISTRY}
