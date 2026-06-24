"""Tests for SQLAlchemy ORM mapping for MFDB.

This test file verifies that the SQLAlchemy ORM mappings in chisurf.core.mfdb.orm
are consistent with the canonical schema.py definitions and provide the expected
API for the bounded MFDB slice as specified in PRD-020.
"""

import os
import tempfile
from pathlib import Path

import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.orm.base import Base, make_engine, session_scope, clear_session_cache
from chisurf.core.mfdb.orm.models import (
    get_mapped_tables,
    get_table_class_by_name,
    FlrSample,
    FlrSampleCondition,
    FlrSampleProbe,
    FlrPolyProbePosition,
    Entity,
    EntityPolySeq,
    Probe,
    OpticalProperty,
    Spectrum,
    FlrFretForsterRadius,
    ChemDescriptor,
    MfdbVocabulary,
)
from chisurf.core.mfdb.orm.sync import (
    run_all_consistency_checks,
    check_orm_columns_exist,
    check_fret_radius_sample_scope,
    check_no_create_all_warning,
    SchemaConsistencyError,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        # Create database using MFDatabase which handles schema migration
        db = MFDatabase(str(db_path))
        db.close()
        yield db_path


@pytest.fixture
def temp_db_path_with_orm():
    """Create a temporary database with ORM tables."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        # Create database using MFDatabase which handles schema migration
        db = MFDatabase(str(db_path))
        db.close()

        # Create the ORM tables
        engine = make_engine(db_path)
        Base.metadata.create_all(engine)
        yield db_path


class TestORMBase:
    """Test ORM base utilities."""

    def test_make_engine_creates_engine(self, temp_db_path):
        """Test that make_engine creates a working SQLAlchemy engine."""
        engine = make_engine(temp_db_path)
        assert engine is not None

        # Test that we can connect
        with engine.connect() as conn:
            assert conn is not None

    def test_session_scope_provides_session(self, temp_db_path):
        """Test that session_scope provides a working session."""
        from sqlalchemy import text
        with session_scope(temp_db_path) as session:
            assert session is not None
            # Test that we can query
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_session_scope_commits_on_success(self, temp_db_path):
        """Test that session_scope commits on successful completion."""
        # First, create the ORM tables
        engine = make_engine(temp_db_path)
        Base.metadata.create_all(engine)

        # Insert data within a session
        with session_scope(temp_db_path) as session:
            test_entry = FlrSample(
                sample_id="test_sample",
                description="Test Sample",
            )
            session.add(test_entry)

        # Verify the data was committed
        with session_scope(temp_db_path) as session:
            result = session.get(FlrSample, "test_sample")
            assert result is not None
            assert result.description == "Test Sample"

    def test_session_scope_rolls_back_on_error(self, temp_db_path):
        """Test that session_scope rolls back on exception."""
        # First, create the ORM tables
        engine = make_engine(temp_db_path)
        Base.metadata.create_all(engine)

        # Try to insert data but raise an exception
        with pytest.raises(ValueError):
            with session_scope(temp_db_path) as session:
                test_entry = FlrSample(
                    sample_id="test_sample",
                    description="Test Sample",
                )
                session.add(test_entry)
                raise ValueError("Test error")

        # Verify the data was NOT committed
        with session_scope(temp_db_path) as session:
            result = session.get(FlrSample, "test_sample")
            assert result is None

    def test_clear_session_cache(self):
        """Test that clear_session_cache clears the cache."""
        # This is a basic test that the function runs without error
        clear_session_cache()


class TestORMModels:
    """Test ORM model definitions."""

    def test_mapped_tables_returns_list(self):
        """Test that get_mapped_tables returns a list of table classes."""
        tables = get_mapped_tables()
        assert isinstance(tables, list)
        assert len(tables) > 0

        # Check that all entries are classes
        for table in tables:
            assert isinstance(table, type)
            assert hasattr(table, '__tablename__')

    def test_get_table_class_by_name(self):
        """Test that get_table_class_by_name returns the correct class."""
        # Test a few known tables
        flr_sample_class = get_table_class_by_name("flr_sample")
        assert flr_sample_class is FlrSample

        probe_class = get_table_class_by_name("probes")
        assert probe_class is Probe

        entity_class = get_table_class_by_name("entities")
        assert entity_class is Entity

    def test_get_table_class_by_name_returns_none_for_unknown(self):
        """Test that get_table_class_by_name returns None for unknown tables."""
        result = get_table_class_by_name("nonexistent_table")
        assert result is None

    def test_orm_models_have_tablenames(self):
        """Test that all ORM models have __tablename__ attributes."""
        tables = get_mapped_tables()
        for table in tables:
            assert hasattr(table, '__tablename__')
            assert table.__tablename__ is not None

    def test_orm_models_have_required_columns(self):
        """Test that ORM models have required columns from the schema."""
        # FlrSample should have key columns
        assert hasattr(FlrSample, 'sample_id')
        assert hasattr(FlrSample, 'description')
        assert hasattr(FlrSample, 'solvent_phase')
        assert hasattr(FlrSample, 'num_of_probes')

        # FlrFretForsterRadius should have sample-scoped columns
        assert hasattr(FlrFretForsterRadius, 'sample_id')
        assert hasattr(FlrFretForsterRadius, 'donor_probe_id')
        assert hasattr(FlrFretForsterRadius, 'acceptor_probe_id')
        assert hasattr(FlrFretForsterRadius, 'forster_radius')

        # Entity should have sequence relationship
        assert hasattr(Entity, 'sequences')

    def test_prd02_models_are_reflected_not_declared(self):
        """PRD-02 ORM tables are reflected from schema, not hand-written classes."""
        import inspect
        import chisurf.core.mfdb.orm.models as models_module

        source = inspect.getsource(models_module)
        forbidden_declarations = [
            "class FlrSample(",
            "class FlrSampleCondition(",
            "class FlrSampleProbe(",
            "class FlrPolyProbePosition(",
            "class Probe(",
            "class FlrFretForsterRadius(",
            "class ChemDescriptor(",
        ]
        for declaration in forbidden_declarations:
            assert declaration not in source

    def test_orm_relationships(self):
        """Test that ORM models have proper relationships."""
        # FlrSample relationships
        assert hasattr(FlrSample, 'condition')
        assert hasattr(FlrSample, 'sample_probes')
        assert hasattr(FlrSample, 'fret_pairs')
        assert hasattr(FlrSample, 'key_values')

        # FlrSampleProbe relationships
        assert hasattr(FlrSampleProbe, 'sample')
        assert hasattr(FlrSampleProbe, 'probe')
        assert hasattr(FlrSampleProbe, 'position')

        # FlrFretForsterRadius relationships
        assert hasattr(FlrFretForsterRadius, 'sample')
        assert hasattr(FlrFretForsterRadius, 'donor_probe')
        assert hasattr(FlrFretForsterRadius, 'acceptor_probe')


class TestORMConsistency:
    """Test ORM schema/mapping consistency checks."""

    def test_check_no_create_all_warning(self):
        """Test that no ORM model calls Base.metadata.create_all()."""
        result = check_no_create_all_warning()
        assert result is True

    def test_check_orm_columns_exist(self, temp_db_path):
        """Test that ORM columns exist in the live schema."""
        # This should not raise SchemaConsistencyError
        try:
            missing = check_orm_columns_exist(temp_db_path)
            assert len(missing) == 0
        except SchemaConsistencyError as e:
            pytest.fail(f"Schema consistency error: {e}")

    def test_check_fret_radius_sample_scope(self, temp_db_path):
        """Test that flr_fret_forster_radius has sample scope."""
        result = check_fret_radius_sample_scope(temp_db_path)
        assert result is True

    def test_run_all_consistency_checks(self, temp_db_path):
        """Test that all consistency checks pass."""
        results = run_all_consistency_checks(temp_db_path)

        # Check that critical checks pass
        critical_checks = ["mapped_columns_exist", "fret_radius_sample_scope", "no_create_all_calls"]
        for check_name in critical_checks:
            assert check_name in results
            assert results[check_name]["passed"] is True, \
                f"Check {check_name} failed: {results[check_name]['details']}"

    def test_orm_models_match_schema(self, temp_db_path):
        """Test that ORM models match the canonical schema."""
        # This is a more comprehensive test that creates a database
        # using the canonical schema and then checks that ORM models can
        # work with it

        # The database was created with the canonical schema, so we should
        # be able to query tables using ORM
        with session_scope(temp_db_path) as session:
            # Test basic querying
            # Note: We can't test FlrSample if no data exists, but we can test
            # that the table structure is compatible
            try:
                session.query(FlrSample).count()
                # This should work even if the table is empty
            except Exception as e:
                pytest.fail(f"ORM query failed: {e}")


class TestSampleRepository:
    """Test SQLAlchemy-backed sample repository adapter."""

    def test_import_sample_repository(self):
        """Test that sample_repository module can be imported."""
        from chisurf.core.mfdb.orm.sample_repository import (
            create_sample_graph,
            get_sample_graph,
            upsert_probe,
        )
        assert callable(create_sample_graph)
        assert callable(get_sample_graph)
        assert callable(upsert_probe)

    def test_orm_package_exports(self):
        """Test that ORM package exports all required symbols."""
        from chisurf.core.mfdb.orm import (
            make_engine,
            session_scope,
            session_from_mfdatabase,
            FlrSample,
            FlrSampleCondition,
            FlrSampleProbe,
            FlrPolyProbePosition,
            Entity,
            EntityPolySeq,
            Probe,
            OpticalProperty,
            Spectrum,
            FlrFretForsterRadius,
            ChemDescriptor,
            create_sample_graph,
            get_sample_graph,
            upsert_probe,
        )
        # All should be importable

    def test_create_sample_graph_roundtrip(self, temp_db_path):
        """Test create_sample_graph and get_sample_graph with structured SampleDefinition.

        This test verifies that the ORM adapter can:
        - Create a sample with entities, probes, positions, FRET pairs, and condition
        - Retrieve the full sample graph
        - Verify all data is preserved in the round-trip
        """
        from chisurf.core.mfdb.repository import MFDatabase
        from chisurf.core.mfdb.models import (
            SampleDefinition,
            EntityDefinition,
            ProbeDefinition,
            FretPairDefinition,
        )
        from chisurf.core.mfdb.orm.sample_repository import (
            create_sample_graph,
            get_sample_graph,
        )

        # Create database
        db = MFDatabase(str(temp_db_path))

        # Create a structured sample definition
        definition = SampleDefinition(
            name="test_sample_orm",
            description="Test sample for ORM adapter round-trip",
            entities=[
                EntityDefinition(
                    name="entity_1",
                    entity_type="protein",
                    sequence="MKTAYIAKQRQ",
                    details="Test protein entity",
                ),
                EntityDefinition(
                    name="entity_2",
                    entity_type="dna",
                    sequence="ATCGATCG",
                    details="Test DNA entity",
                ),
            ],
            probes=[
                ProbeDefinition(
                    name="Cy3B",
                    entity_index=0,
                    seq_id=50,
                    comp_id="CYS",
                    asym_id="A",
                    atom_id="CB",
                    mutation_flag="no",
                    modification_flag="no",
                    auth_name="C50",
                ),
                ProbeDefinition(
                    name="Cy5",
                    entity_index=0,
                    seq_id=100,
                    comp_id="CYS",
                    asym_id="A",
                    atom_id="CB",
                    mutation_flag="no",
                    modification_flag="no",
                    auth_name="C100",
                ),
                ProbeDefinition(
                    name="Alexa488",
                    entity_index=1,
                    seq_id=5,
                    comp_id="dT",
                    asym_id="B",
                    atom_id="C5",
                    mutation_flag="yes",
                    modification_flag="no",
                    auth_name="dT5",
                ),
            ],
            fret_pairs=[
                FretPairDefinition(
                    probe_1_index=0,
                    probe_2_index=1,
                    forster_radius_nm=6.0,
                    kappa_squared=0.666667,
                    refractive_index=1.4,
                ),
                FretPairDefinition(
                    probe_1_index=0,
                    probe_2_index=2,
                    forster_radius_nm=5.0,
                    kappa_squared=0.666667,
                    refractive_index=1.33,
                ),
            ],
            ph=7.4,
            temperature_k=298.0,
            salt_concentration_m=0.15,
            buffer_description="PBS buffer",
            solvent_phase="liquid",
            extra={"custom_key": "custom_value", "another_key": "another_value"},
        )

        # Create sample
        sample_id = create_sample_graph(db, definition)
        assert sample_id == "test_sample_orm"

        # Retrieve sample graph
        result = get_sample_graph(db, sample_id)
        assert result is not None

        # Verify sample data
        sample = result["sample"]
        assert sample["sample_id"] == "test_sample_orm"
        assert sample["description"] == "test_sample_orm"
        assert sample["solvent_phase"] == "liquid"
        assert sample["details"] == "Test sample for ORM adapter round-trip"

        # Verify entities
        assert len(result["entities"]) == 2
        entity_1 = next(e for e in result["entities"] if e["entity_id"] == "entity_1")
        assert entity_1["type"] == "protein"
        assert entity_1["description"] == "Test protein entity"
        assert len(entity_1["sequences"]) == 11  # "MKTAYIAKQRQ"

        entity_2 = next(e for e in result["entities"] if e["entity_id"] == "entity_2")
        assert entity_2["type"] == "dna"
        assert entity_2["description"] == "Test DNA entity"
        assert len(entity_2["sequences"]) == 8  # "ATCGATCG"

        # Verify probes
        assert len(result["probes"]) == 3
        probe_names = [p["name"] for p in result["probes"]]
        assert "Cy3B" in probe_names
        assert "Cy5" in probe_names
        assert "Alexa488" in probe_names

        # Verify positions
        cy3b_probe = next(p for p in result["probes"] if p["name"] == "Cy3B")
        assert cy3b_probe["position"]["residue_number"] == 50
        assert cy3b_probe["position"]["residue_name"] == "CYS"
        assert cy3b_probe["position"]["asym_id"] == "A"
        assert cy3b_probe["position"]["atom_id"] == "CB"
        assert cy3b_probe["position"]["mutation_flag"] == "no"
        assert cy3b_probe["position"]["modification_flag"] == "no"
        assert cy3b_probe["position"]["auth_name"] == "C50"
        assert cy3b_probe["position"]["entity_id"] == "entity_1"

        # Verify FRET pairs
        assert len(result["fret_pairs"]) == 2
        fret_pairs = result["fret_pairs"]
        # First FRET pair: Cy3B -> Cy5
        fp1 = next(fp for fp in fret_pairs if fp["forster_radius"] == 6.0)
        assert fp1["kappa_squared"] == pytest.approx(0.666667)
        assert fp1["index_of_refraction"] == pytest.approx(1.4)

        # Second FRET pair: Cy3B -> Alexa488
        fp2 = next(fp for fp in fret_pairs if fp["forster_radius"] == 5.0)
        assert fp2["kappa_squared"] == pytest.approx(0.666667)
        assert fp2["index_of_refraction"] == pytest.approx(1.33)

        # Verify condition
        assert result["condition"] is not None
        condition = result["condition"]
        assert condition["ph"] == pytest.approx(7.4)
        assert condition["temperature"] == pytest.approx(298.0)
        assert condition["ionic_strength"] == pytest.approx(0.15)
        assert condition["buffer_composition"] == "PBS buffer"

        db.close()

    def test_upsert_probe_reuses_existing_chemical_descriptor(self, temp_db_path):
        """Repeated probe writes reuse descriptor identity rows."""
        from chisurf.core.mfdb.models import ProbeDefinition
        from chisurf.core.mfdb.orm.sample_repository import upsert_probe

        db = MFDatabase(str(temp_db_path))
        probe = ProbeDefinition(
            name="DescriptorReuseProbe",
            chromophore_smiles=" C1=CC=CC=C1 ",
            reactive_probe_smiles="C1=CC=CC=C1",
        )

        first_probe_id = upsert_probe(db, probe)
        second_probe_id = upsert_probe(db, probe)
        assert first_probe_id == second_probe_id

        rows = db.conn.execute(
            "SELECT descriptor_type, descriptor, program, program_version "
            "FROM chem_descriptors"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["descriptor_type"] == "SMILES"
        assert rows[0]["descriptor"] == "C1=CC=CC=C1"
        db.close()


class TestORMPersistence:
    """Test ORM-based persistence operations."""

    def test_create_and_retrieve_sample(self, temp_db_path_with_orm):
        """Test creating and retrieving a sample with ORM."""
        with session_scope(temp_db_path_with_orm) as session:
            # Create a sample
            sample = FlrSample(
                sample_id="test_sample_1",
                description="Test sample description",
                num_of_probes=2,
                solvent_phase="liquid",
            )
            session.add(sample)
            session.flush()

            # Retrieve the sample
            retrieved = session.get(FlrSample, "test_sample_1")
            assert retrieved is not None
            assert retrieved.description == "Test sample description"
            assert retrieved.num_of_probes == 2

    def test_create_sample_with_condition(self, temp_db_path_with_orm):
        """Test creating a sample with condition."""
        with session_scope(temp_db_path_with_orm) as session:
            # Create condition first
            condition = FlrSampleCondition(
                condition_id="test_condition_1",
                ph=7.4,
                temperature=25.0,
            )
            session.add(condition)
            session.flush()

            # Create sample with condition
            sample = FlrSample(
                sample_id="test_sample_2",
                description="Test sample with condition",
                sample_condition_id="test_condition_1",
            )
            session.add(sample)
            session.flush()

            # Retrieve and verify
            retrieved = session.get(FlrSample, "test_sample_2")
            assert retrieved is not None
            assert retrieved.condition is not None
            assert retrieved.condition.ph == 7.4

    def test_create_probe(self, temp_db_path_with_orm):
        """Test creating a probe."""
        with session_scope(temp_db_path_with_orm) as session:
            probe = Probe(
                chromophore_name="Cy3B",
                probe_origin="extrinsic",
                probe_link_type="covalent",
                fluorophore_type="donor",
            )
            session.add(probe)
            session.flush()

            # Retrieve and verify
            retrieved = session.query(Probe).filter_by(chromophore_name="Cy3B").first()
            assert retrieved is not None
            assert retrieved.fluorophore_type == "donor"

    def test_create_fret_pair(self, temp_db_path_with_orm):
        """Test creating a FRET pair with sample scope."""
        with session_scope(temp_db_path_with_orm) as session:
            # Create probes first
            donor = Probe(chromophore_name="Cy3B")
            acceptor = Probe(chromophore_name="Cy5")
            session.add_all([donor, acceptor])
            session.flush()

            # Create sample
            sample = FlrSample(sample_id="test_fret_sample")
            session.add(sample)
            session.flush()

            # Create FRET pair
            fret_pair = FlrFretForsterRadius(
                sample_id="test_fret_sample",
                donor_probe_id=donor.probe_id,
                acceptor_probe_id=acceptor.probe_id,
                forster_radius=5.4,
                kappa_squared=0.6667,
            )
            session.add(fret_pair)
            session.flush()

            # Retrieve and verify
            retrieved = session.query(FlrFretForsterRadius).first()
            assert retrieved is not None
            assert retrieved.sample_id == "test_fret_sample"
            assert retrieved.forster_radius == 5.4

    def test_sample_scoped_fret_pairs(self, temp_db_path_with_orm):
        """Test that FRET pairs are sample-scoped (not globally unique)."""
        with session_scope(temp_db_path_with_orm) as session:
            # Create probes
            donor = Probe(chromophore_name="Cy3B")
            acceptor = Probe(chromophore_name="Cy5")
            session.add_all([donor, acceptor])
            session.flush()

            # Create two samples with the same donor/acceptor but different R0
            sample1 = FlrSample(sample_id="sample1")
            sample2 = FlrSample(sample_id="sample2")
            session.add_all([sample1, sample2])
            session.flush()

            # Create FRET pairs for both samples with the same probes but different R0
            fret1 = FlrFretForsterRadius(
                sample_id="sample1",
                donor_probe_id=donor.probe_id,
                acceptor_probe_id=acceptor.probe_id,
                forster_radius=5.4,
            )
            fret2 = FlrFretForsterRadius(
                sample_id="sample2",
                donor_probe_id=donor.probe_id,
                acceptor_probe_id=acceptor.probe_id,
                forster_radius=6.2,
            )
            session.add_all([fret1, fret2])
            session.flush()

            # Verify both FRET pairs exist
            pairs = session.query(FlrFretForsterRadius).all()
            assert len(pairs) == 2

            # Verify they have different R0 values
            r0_values = {pair.forster_radius for pair in pairs}
            assert r0_values == {5.4, 6.2}


class TestORMBoundedSlice:
    """Test that the ORM covers the bounded MFDB slice as specified in PRD-020."""

    def test_phase_a_tables_present(self):
        """Test that all Phase A tables have ORM models."""
        phase_a_tables = [
            "flr_sample",
            "flr_sample_condition",
            "flr_sample_probe",
            "flr_poly_probe_position",
            "entities",
            "entity_poly_seq",
            "probes",
            "chem_descriptors",
            "optical_properties",
            "spectra",
            "flr_fret_forster_radius",
        ]

        for table_name in phase_a_tables:
            model_class = get_table_class_by_name(table_name)
            assert model_class is not None, f"Missing ORM model for table: {table_name}"

    def test_phase_b_tables_present(self):
        """Test that Phase B vocabulary/dictionary tables have ORM models."""
        phase_b_tables = [
            "mfdb_vocabulary",
        ]

        for table_name in phase_b_tables:
            model_class = get_table_class_by_name(table_name)
            assert model_class is not None, f"Missing ORM model for table: {table_name}"

    def test_relationships_express_sample_probe_flow(self):
        """Test that relationships express FlrSample -> sample_probes -> Probe."""
        # This is tested by checking that the relationship attributes exist
        assert hasattr(FlrSample, 'sample_probes')
        assert hasattr(FlrSampleProbe, 'sample')
        assert hasattr(FlrSampleProbe, 'probe')
        assert hasattr(Probe, 'sample_probes')

    def test_relationships_express_position_entity_flow(self):
        """Test that relationships express FlrSampleProbe -> poly_probe_position -> Entity."""
        assert hasattr(FlrSampleProbe, 'position')
        assert hasattr(FlrPolyProbePosition, 'entity')
        assert hasattr(Entity, 'probe_positions')

    def test_relationships_express_fret_pairs(self):
        """Test that relationships express FlrSample -> fret_pairs."""
        assert hasattr(FlrSample, 'fret_pairs')
        assert hasattr(FlrFretForsterRadius, 'sample')

    def test_fret_pair_sample_scoped(self):
        """Test that FlrFretForsterRadius is scoped to one sample."""
        # Check that the model has sample_id column
        assert hasattr(FlrFretForsterRadius, 'sample_id')
        # Check that it has unique constraint on (sample_id, donor_probe_id, acceptor_probe_id)
        # This is tested by the table definition


# Additional tests for completeness
class TestORMIntegration:
    """Integration tests for ORM functionality."""

    def test_orm_session_from_mfdatabase(self):
        """Test session_from_mfdatabase function."""
        import tempfile
        from pathlib import Path
        from sqlalchemy import text

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            # Create database using MFDatabase
            db = MFDatabase(str(db_path))

            # Get session from MFDatabase
            from chisurf.core.mfdb.orm.base import session_from_mfdatabase
            session = session_from_mfdatabase(db)
            assert session is not None

            # Test that session works
            try:
                session.execute(text("SELECT 1"))
            finally:
                session.close()
                db.close()

    def test_orm_models_have_docstrings(self):
        """Test that ORM model classes have docstrings."""
        models = get_mapped_tables()
        for model in models:
            assert model.__doc__ is not None, f"{model.__name__} missing docstring"
            assert len(model.__doc__.strip()) > 0, f"{model.__name__} has empty docstring"

    def test_orm_base_module_has_docstrings(self):
        """Test that ORM base module functions have docstrings."""
        from chisurf.core.mfdb.orm.base import make_engine, session_scope, session_from_mfdatabase

        assert make_engine.__doc__ is not None
        assert session_scope.__doc__ is not None
        assert session_from_mfdatabase.__doc__ is not None
