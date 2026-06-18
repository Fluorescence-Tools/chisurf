"""Request data classes for sample operations.

This module provides canonical input objects for sample-related operations,
following the pattern established by other plugins (burst_selection, fret, etc.).
These Request classes are used as input to API endpoints, RPC handlers, and
GUI dialogs, ensuring type safety and consistent validation.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from chisurf.core.mfdb.models import (
    COMMON_PROBE_NAMES,
    ENTITY_TYPES,
    BUFFER_COMPONENTS,
    SAMPLE_CONDITION_FIELDS,
    SampleDefinition,
    EntityDefinition,
    ProbeDefinition,
    FretPairDefinition,
    validate_vocabulary,
)


@dataclass
class SampleCreateRequest:
    """Request to create a new sample.

    This is the canonical input for sample creation through API, RPC, and GUI.
    All fields are optional except ``name``.

    Attributes
    ----------
    name : str
        Human-readable sample name (required).
    description : str, optional
        Free-text sample description.
    entities : list of EntityDefinition, optional
        List of biomolecular entities in the sample (PRD-02).
        For backward compatibility, entity_name/entity_type/entity_sequence
        can also be used for single-entity samples.
    entity_name : str, optional
        Entity common name, for example ``"T4 lysozyme"``.
        Legacy field for backward compatibility.
    entity_sequence : str, optional
        Amino-acid or nucleotide sequence.
        Legacy field for backward compatibility.
    entity_type : str, optional
        Entity type, must be one of ``ENTITY_TYPES`` if provided.
        Legacy field for backward compatibility.
    probes : list of ProbeDefinition, optional
        List of all probes on the sample (PRD-02).
        For backward compatibility, donor_probe_name/acceptor_probe_name
        can also be used for single-pair samples.
    fret_pairs : list of FretPairDefinition, optional
        List of FRET pairs defining energy transfer relationships (PRD-02).
    donor_probe_name : str, optional
        Donor probe name, must be in ``COMMON_PROBE_NAMES`` if provided.
        Legacy field for backward compatibility.
    donor_position : int, optional
        Donor residue or sequence position.
        Legacy field for backward compatibility.
    donor_position_label : str, optional
        Human-readable donor position label.
        Legacy field for backward compatibility.
    acceptor_probe_name : str, optional
        Acceptor probe name, must be in ``COMMON_PROBE_NAMES`` if provided.
        Legacy field for backward compatibility.
    acceptor_position : int, optional
        Acceptor residue or sequence position.
        Legacy field for backward compatibility.
    acceptor_position_label : str, optional
        Human-readable acceptor position label.
        Legacy field for backward compatibility.
    buffer_description : str, optional
        Buffer description, for example ``"PBS pH 7.4"``.
    ph : float, optional
        Sample pH.
    temperature_k : float, optional
        Temperature in Kelvin.
    salt_concentration_m : float, optional
        Salt concentration in mol/L.
    extra : dict, optional
        Additional metadata stored as JSON.
    validate_vocabulary : bool, optional
        If ``True`` (default), validate entity_type and probe names against
        known vocabularies.

    """

    name: str
    description: str = ""
    # New structured fields for multi-entity/multi-probe support (PRD-02)
    entities: list[EntityDefinition] = field(default_factory=list)
    probes: list[ProbeDefinition] = field(default_factory=list)
    fret_pairs: list[FretPairDefinition] = field(default_factory=list)
    # Legacy flat fields for backward compatibility
    entity_name: str = ""
    entity_sequence: str = ""
    entity_type: str = ""
    donor_probe_name: str = ""
    donor_position: Optional[int] = None
    donor_position_label: str = ""
    acceptor_probe_name: str = ""
    acceptor_position: Optional[int] = None
    acceptor_position_label: str = ""
    buffer_description: str = ""
    ph: Optional[float] = None
    temperature_k: Optional[float] = None
    salt_concentration_m: Optional[float] = None
    extra: dict = field(default_factory=dict)
    validate_vocabulary: bool = True  # R12-6: renamed for consistency with SampleDefinition

    def __post_init__(self) -> None:
        """Validate vocabulary fields if validation is enabled."""
        if not self.name or not self.name.strip():
            raise ValueError("sample name is required")

        if self.validate_vocabulary:
            self._validate_vocabulary()

    def _validate_vocabulary(self) -> None:
        """Validate entity_type and probe names against known vocabularies."""
        # Validate entity_type from legacy field
        if self.entity_type:
            validate_vocabulary(
                self.entity_type,
                ENTITY_TYPES,
                "entity_type"
            )
        
        # Validate entity_types from new entities list
        for i, entity in enumerate(self.entities):
            if entity.entity_type:
                validate_vocabulary(
                    entity.entity_type,
                    ENTITY_TYPES,
                    f"entities[{i}].entity_type"
                )

        # Validate probe names from legacy fields
        if self.donor_probe_name:
            validate_vocabulary(
                self.donor_probe_name,
                COMMON_PROBE_NAMES,
                "donor_probe_name"
            )

        if self.acceptor_probe_name:
            validate_vocabulary(
                self.acceptor_probe_name,
                COMMON_PROBE_NAMES,
                "acceptor_probe_name"
            )
        
        # Validate probe names from new probes list
        for i, probe in enumerate(self.probes):
            if probe.name:
                validate_vocabulary(
                    probe.name,
                    COMMON_PROBE_NAMES,
                    f"probes[{i}].name"
                )

    def to_sample_definition(self) -> SampleDefinition:
        """Convert this request to a ``SampleDefinition``.

        Returns
        -------
        SampleDefinition
            Equivalent sample definition for use with sample_manager functions.
            
        Notes
        -----
        If both new structured fields (entities, probes, fret_pairs) and legacy
        flat fields are provided, the new fields take precedence.

        """
        # Use new structured fields if provided, fall back to legacy fields
        entities = self.entities if self.entities else None
        probes = self.probes if self.probes else None
        fret_pairs = self.fret_pairs if self.fret_pairs else None
        
        return SampleDefinition(
            name=self.name,
            description=self.description,
            entities=entities or [],
            entity_name=self.entity_name,
            entity_sequence=self.entity_sequence,
            entity_type=self.entity_type,
            probes=probes or [],
            fret_pairs=fret_pairs or [],
            donor_probe_name=self.donor_probe_name,
            donor_position=self.donor_position,
            donor_position_label=self.donor_position_label,
            acceptor_probe_name=self.acceptor_probe_name,
            acceptor_position=self.acceptor_position,
            acceptor_position_label=self.acceptor_position_label,
            buffer_description=self.buffer_description,
            ph=self.ph,
            temperature_k=self.temperature_k,
            salt_concentration_m=self.salt_concentration_m,
            extra=self.extra,
            validate_vocabulary=self.validate_vocabulary,
        )


@dataclass
class SampleUpdateRequest:
    """Request to update an existing sample.

    Attributes
    ----------
    sample_id : str
        The sample ID to update (required).
    name : str, optional
        New display name. If not provided, the existing name is kept.
    description : str, optional
        New description. If not provided, the existing description is kept.
    entities : list of EntityDefinition, optional
        New list of biomolecular entities (PRD-02).
    entity_name : str, optional
        New entity common name (legacy field).
    entity_sequence : str, optional
        New entity sequence (legacy field).
    entity_type : str, optional
        New entity type, validated against ``ENTITY_TYPES`` (legacy field).
    probes : list of ProbeDefinition, optional
        New list of all probes on the sample (PRD-02).
    fret_pairs : list of FretPairDefinition, optional
        New list of FRET pairs (PRD-02).
    donor_probe_name : str, optional
        New donor probe name, validated against ``COMMON_PROBE_NAMES`` (legacy field).
    donor_position : int, optional
        New donor position (legacy field).
    donor_position_label : str, optional
        New donor position label (legacy field).
    acceptor_probe_name : str, optional
        New acceptor probe name, validated against ``COMMON_PROBE_NAMES`` (legacy field).
    acceptor_position : int, optional
        New acceptor position (legacy field).
    acceptor_position_label : str, optional
        New acceptor position label (legacy field).
    buffer_description : str, optional
        New buffer description.
    ph : float, optional
        New pH value.
    temperature_k : float, optional
        New temperature in Kelvin.
    salt_concentration_m : float, optional
        New salt concentration in mol/L.
    extra : dict, optional
        Additional metadata to merge into existing extra.
    validate_vocabulary : bool, optional
        If ``True`` (default), validate entity_type and probe names.

    """

    sample_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    # New structured fields for PRD-02
    entities: Optional[list[EntityDefinition]] = None
    probes: Optional[list[ProbeDefinition]] = None
    fret_pairs: Optional[list[FretPairDefinition]] = None
    # Legacy flat fields for backward compatibility
    entity_name: Optional[str] = None
    entity_sequence: Optional[str] = None
    entity_type: Optional[str] = None
    donor_probe_name: Optional[str] = None
    donor_position: Optional[int] = None
    donor_position_label: Optional[str] = None
    acceptor_probe_name: Optional[str] = None
    acceptor_position: Optional[int] = None
    acceptor_position_label: Optional[str] = None
    buffer_description: Optional[str] = None
    ph: Optional[float] = None
    temperature_k: Optional[float] = None
    salt_concentration_m: Optional[float] = None
    extra: dict = field(default_factory=dict)
    validate_vocabulary: bool = True  # R12-6/R13-4: renamed for consistency

    def __post_init__(self) -> None:
        """Validate vocabulary fields if validation is enabled."""
        if not self.sample_id:
            raise ValueError("sample_id is required for update")

        if self.validate_vocabulary:
            self._validate_vocabulary()

    def _validate_vocabulary(self) -> None:
        """Validate entity_type and probe names against known vocabularies."""
        # Validate entity_type from legacy field
        if self.entity_type:
            validate_vocabulary(
                self.entity_type,
                ENTITY_TYPES,
                "entity_type"
            )
        
        # Validate entity_types from new entities list
        if self.entities:
            for i, entity in enumerate(self.entities):
                if entity.entity_type:
                    validate_vocabulary(
                        entity.entity_type,
                        ENTITY_TYPES,
                        f"entities[{i}].entity_type"
                    )

        # Validate probe names from legacy fields
        if self.donor_probe_name:
            validate_vocabulary(
                self.donor_probe_name,
                COMMON_PROBE_NAMES,
                "donor_probe_name"
            )

        if self.acceptor_probe_name:
            validate_vocabulary(
                self.acceptor_probe_name,
                COMMON_PROBE_NAMES,
                "acceptor_probe_name"
            )
        
        # Validate probe names from new probes list
        if self.probes:
            for i, probe in enumerate(self.probes):
                if probe.name:
                    validate_vocabulary(
                        probe.name,
                        COMMON_PROBE_NAMES,
                        f"probes[{i}].name"
                    )


@dataclass
class SampleQueryRequest:
    """Request to query samples with filters.

    Attributes
    ----------
    name_contains : str, optional
        Filter by display name containing this substring (case-insensitive).
    entity_type : str, optional
        Filter by entity type.
    probe_name : str, optional
        Filter by probe name (donor or acceptor).
    ph_min : float, optional
        Minimum pH value for filtering.
    ph_max : float, optional
        Maximum pH value for filtering.
    temperature_min : float, optional
        Minimum temperature in Kelvin for filtering.
    temperature_max : float, optional
        Maximum temperature in Kelvin for filtering.
    limit : int, optional
        Maximum number of results to return. Default is 100.
    offset : int, optional
        Number of results to skip. Default is 0.

    """

    name_contains: Optional[str] = None
    entity_type: Optional[str] = None
    probe_name: Optional[str] = None
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    limit: int = 100
    offset: int = 0

    def __post_init__(self) -> None:
        """Validate vocabulary fields."""
        if self.entity_type:
            validate_vocabulary(
                self.entity_type,
                ENTITY_TYPES,
                "entity_type"
            )

        if self.probe_name:
            validate_vocabulary(
                self.probe_name,
                COMMON_PROBE_NAMES,
                "probe_name"
            )

        if self.limit < 1:
            raise ValueError("limit must be at least 1")

        if self.offset < 0:
            raise ValueError("offset must be non-negative")


@dataclass
class SampleLinkRequest:
    """Request to link an artifact to a sample.

    Attributes
    ----------
    artifact_id : str
        The artifact ID to link.
    sample_id : str
        The sample ID to link to.

    """

    artifact_id: str
    sample_id: str

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.artifact_id:
            raise ValueError("artifact_id is required")

        if not self.sample_id:
            raise ValueError("sample_id is required")


@dataclass
class SampleUnlinkRequest:
    """Request to unlink an artifact from a sample.

    Attributes
    ----------
    artifact_id : str
        The artifact ID to unlink.
    sample_id : str
        The sample ID to unlink from.

    """

    artifact_id: str
    sample_id: str

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.artifact_id:
            raise ValueError("artifact_id is required")

        if not self.sample_id:
            raise ValueError("sample_id is required")


@dataclass
class SampleSearchRequest:
    """Request to search samples by PDBx/mmCIF/flrCIF vocabulary keys.

    This request allows searching samples using standardized vocabulary keys
    from PDBx, pdbihm, or flrCIF dictionaries.

    Attributes
    ----------
    vocabulary_field : str, optional
        The vocabulary field name to search (e.g., "entity_type", "ph").
    vocabulary_value : str, optional
        The value to match against the vocabulary field.
    use_pdbx : bool, optional
        If ``True``, validate against PDBx dictionary keys. Default is ``True``.
    use_flrcif : bool, optional
        If ``True``, validate against flrCIF dictionary keys. Default is ``True``.

    """

    vocabulary_field: Optional[str] = None
    vocabulary_value: Optional[str] = None
    use_pdbx: bool = True
    use_flrcif: bool = True

    def __post_init__(self) -> None:
        """Validate vocabulary field."""
        if self.vocabulary_field:
            valid_fields = set()
            
            if self.use_pdbx:
                try:
                    from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
                    dic = MmcifDictionary.load_bundled()
                    # Get all category.field keys from the dictionary
                    for category_name, category in dic._categories.items():
                        if hasattr(category, 'items') and category.items:
                            for field_name in category.items.keys():
                                valid_fields.add(f"{category_name}.{field_name}")
                except Exception:
                    pass  # Dictionary not available, skip PDBx validation
            
            if self.use_flrcif:
                try:
                    from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
                    dic = MmcifDictionary.load_bundled()
                    # flrCIF uses the same dictionary, so include all flr_* categories
                    for category_name, category in dic._categories.items():
                        if category_name.startswith("flr_") and hasattr(category, 'items') and category.items:
                            for field_name in category.items.keys():
                                valid_fields.add(f"{category_name}.{field_name}")
                except Exception:
                    pass  # Dictionary not available, skip flrCIF validation
            
            # Check if the field is valid
            if valid_fields:
                # Allow partial matches for flexibility
                matching_fields = [f for f in valid_fields if self.vocabulary_field in f]
                if not matching_fields:
                    raise ValueError(
                        f"vocabulary_field '{self.vocabulary_field}' not found in vocabulary. "
                        f"Valid fields include: {', '.join(sorted(valid_fields)[:10])}..."
                    )
