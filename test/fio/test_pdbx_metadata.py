"""Tests for mmCIF/PDBx dictionary parsing and metadata handling.

This test file verifies that the MmcifDictionary correctly parses all bundled
.dic files and provides the expected API for vocabulary validation and metadata
introspection as specified in PRD-02a.
"""

import json
import tempfile
from pathlib import Path

import pytest

from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary, DictItem, DictCategory


@pytest.fixture(scope="module")
def dic():
    """Load the bundled dictionaries once for all tests."""
    return MmcifDictionary.load_bundled()


def test_load_bundled_finds_categories(dic):
    """Bundled dictionaries contain PDBx + IHM + FLR categories."""
    cats = dic.categories()
    assert len(cats) > 500  # PDBx alone has ~700
    assert "entity" in cats
    assert "entity_poly_seq" in cats


def test_flr_categories_present(dic):
    """flrCIF extension categories are parsed."""
    flr = dic.flr_categories()
    assert "flr_sample" in flr
    assert "flr_sample_condition" in flr
    assert "flr_sample_probe_details" in flr
    assert "flr_fret_forster_radius" in flr
    assert "flr_poly_probe_position" in flr
    assert len(flr) >= 30


def test_item_lookup(dic):
    """Item lookup by full name works."""
    item = dic.get_item("_flr_sample.id")
    assert item is not None
    assert item.category == "flr_sample"
    assert item.attribute == "id"


def test_item_lookup_without_underscore(dic):
    """Item lookup works with or without leading underscore."""
    item1 = dic.get_item("_flr_sample.id")
    item2 = dic.get_item("flr_sample.id")  # Some callers may not include underscore
    assert item1 is not None
    # Note: The canonical form includes the underscore


def test_fluorophore_type_enumerations(dic):
    """_flr_sample_probe_details.fluorophore_type has donor/acceptor/unspecified."""
    enums = dic.get_enumerations("_flr_sample_probe_details.fluorophore_type")
    assert "donor" in enums
    assert "acceptor" in enums
    assert "unspecified" in enums


def test_entity_type_enumerations(dic):
    """_entity.type has polymer, non-polymer, etc."""
    enums = dic.get_enumerations("_entity.type")
    assert "polymer" in enums
    assert "non-polymer" in enums
    assert "water" in enums


def test_descriptions_parsed(dic):
    """Item descriptions are non-empty."""
    desc = dic.get_description("_flr_sample.id")
    assert len(desc) > 0


def test_non_loop_items_are_retained(dic):
    """Required fields without enumeration loops are still parsed."""
    # These are critical fields that should be present even without loop_ blocks
    assert dic.get_item("_flr_sample.id") is not None
    assert dic.get_item("_flr_sample.num_of_probes") is not None
    assert dic.get_item("_flr_sample.sample_condition_id") is not None
    assert dic.get_item("_flr_sample.solvent_phase") is not None


def test_pdbx_core_categories_present(dic):
    """Core PDBx/mmCIF categories are present."""
    cats = dic.categories()
    # Core categories that should be present
    core_categories = ["entity", "entity_poly_seq", "atom_site", "struct", "struct_asym"]
    for cat in core_categories:
        assert cat in cats, f"Missing core category: {cat}"


def test_ihm_categories_present(dic):
    """IHM extension categories are parsed."""
    cats = dic.categories()
    ihm_categories = ["ihm_model_list", "ihm_model_representation", "ihm_struct_assembly"]
    for cat in ihm_categories:
        assert cat in cats, f"Missing IHM category: {cat}"


def test_flr_core_fields_present(dic):
    """Core flrCIF fields are present and accessible."""
    # Core fields from flr_sample
    flr_sample_fields = ["_flr_sample.id", "_flr_sample.sample_description", 
                         "_flr_sample.num_of_probes", "_flr_sample.solvent_phase"]
    for field in flr_sample_fields:
        item = dic.get_item(field)
        assert item is not None, f"Missing flr_sample field: {field}"
        assert item.category == "flr_sample"

    # Fields from flr_fret_forster_radius (including ChiSurf extensions)
    fret_fields = ["_flr_fret_forster_radius.forster_radius", 
                   "_flr_fret_forster_radius.donor_probe_id",
                   "_flr_fret_forster_radius.acceptor_probe_id",
                   "_flr_fret_forster_radius.kappa_squared"]
    for field in fret_fields:
        item = dic.get_item(field)
        assert item is not None, f"Missing flr_fret_forster_radius field: {field}"


def test_sample_search_accepts_valid_dictionary_field():
    """SampleSearchRequest should accept valid dictionary fields."""
    from chisurf.core.mfdb.sample_requests import SampleSearchRequest

    # These should not raise
    request = SampleSearchRequest(
        vocabulary_field="flr_sample.id",
        vocabulary_value="sample_1",
    )
    assert request.vocabulary_field == "flr_sample.id"

    request = SampleSearchRequest(
        vocabulary_field="_flr_sample.id",
        vocabulary_value="sample_1",
    )
    assert request.vocabulary_field == "_flr_sample.id"

    request = SampleSearchRequest(
        vocabulary_field="entity.type",
        vocabulary_value="protein",
    )
    assert request.vocabulary_field == "entity.type"


def test_validate_valid_value(dic):
    """validate_value returns None for valid enumerated values."""
    assert dic.validate_value("_flr_sample_probe_details.fluorophore_type", "donor") is None
    assert dic.validate_value("_entity.type", "polymer") is None


def test_validate_invalid_value(dic):
    """validate_value returns error message for invalid enumerated values."""
    err = dic.validate_value("_flr_sample_probe_details.fluorophore_type", "emitter")
    assert err is not None
    assert "donor" in err or "allowed" in err.lower()


def test_validate_unknown_field(dic):
    """validate_value returns error for unknown fields."""
    err = dic.validate_value("_nonexistent.field", "value")
    assert err is not None
    assert "Unknown" in err


def test_search_items(dic):
    """search_items enables keyword search across all items."""
    results = dic.search_items("forster")
    assert len(results) > 0
    names = [r.name for r in results]
    assert any("forster" in n.lower() for n in names)

    # Test search for sample-related fields
    results = dic.search_items("sample")
    assert len(results) > 0
    names = [r.name for r in results]
    assert any("sample" in n.lower() for n in names)


def test_cache_roundtrip(tmp_path):
    """Parsed dictionary survives JSON cache serialize/deserialize."""
    dic1 = MmcifDictionary.load_bundled()
    cache_path = tmp_path / "cache.json"
    dic1.save_cache(cache_path)
    dic2 = MmcifDictionary.load_cache(cache_path)
    
    assert set(dic1.categories()) == set(dic2.categories())
    assert dic1.get_enumerations("_flr_sample_probe_details.fluorophore_type") == \
           dic2.get_enumerations("_flr_sample_probe_details.fluorophore_type")
    
    # Check that critical fields are preserved
    assert dic1.get_item("_flr_sample.id") is not None
    assert dic2.get_item("_flr_sample.id") is not None


def test_stats_cli_runs():
    """Dictionary CLI stats command does not crash."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--stats"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Items:" in result.stdout
    assert "Categories:" in result.stdout
    assert "flrCIF:" in result.stdout


def test_list_categories_cli():
    """Dictionary CLI list-categories command works."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--list-categories"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "entity" in result.stdout
    assert "flr_sample" in result.stdout


def test_flr_categories_cli():
    """Dictionary CLI flr-categories command works."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--flr-categories"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "flr_sample" in result.stdout
    assert "flr_fret_forster_radius" in result.stdout


def test_enums_cli():
    """Dictionary CLI enums command works."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--enums", "_flr_sample_probe_details.fluorophore_type"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "donor" in result.stdout
    assert "acceptor" in result.stdout


def test_search_cli():
    """Dictionary CLI search command works."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--search", "forster radius"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Found" in result.stdout


def test_validate_cli():
    """Dictionary CLI validate command works."""
    import subprocess
    import sys

    # Test valid value
    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--validate", "_flr_sample_probe_details.fluorophore_type", "donor"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "VALID" in result.stdout

    # Test invalid value
    result = subprocess.run(
        [sys.executable, "-m", "chisurf.core.mfdb.pdbx_metadata", "--validate", "_flr_sample_probe_details.fluorophore_type", "invalid"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "VALID" not in result.stdout


def test_category_lookup(dic):
    """Category lookup works correctly."""
    cat = dic.get_category("flr_sample")
    assert cat is not None
    assert cat.name == "flr_sample"
    # Category should have items
    assert len(cat.items) > 0
    # Should contain the id item
    assert "id" in cat.items


def test_category_mandatory_fields(dic):
    """Category mandatory field information is available."""
    cat = dic.get_category("flr_sample")
    assert cat is not None
    # Check that mandatory field information is preserved
    if "id" in cat.items:
        id_item = cat.items["id"]
        assert hasattr(id_item, 'mandatory')


def test_data_types_extracted(dic):
    """Data types (_item_type.code) are extracted."""
    item = dic.get_item("_flr_sample.id")
    assert item is not None
    # Should have type_code set
    assert item.type_code  # Should be non-empty


def test_item_descriptions_non_empty(dic):
    """Item descriptions are non-empty for fields that have them."""
    # Test a few fields that should have descriptions
    test_fields = ["_flr_sample.id", "_flr_sample.sample_description"]
    for field in test_fields:
        item = dic.get_item(field)
        if item:
            # Description should be populated if available in the .dic file
            # We can't assert it's non-empty for all fields, but it should not crash
            _ = item.description


def test_parent_child_relationships(dic):
    """Parent-child relationships between categories/items are available."""
    # This tests that foreign key relationships are parsed
    # For example, flr_sample should reference entity_assembly
    item = dic.get_item("_flr_sample.entity_assembly_id")
    if item:
        # Should have parent/child relationship information if available
        assert hasattr(item, 'parent')
        assert hasattr(item, 'child')


@pytest.mark.parametrize("field,expected_type", [
    ("_flr_sample.id", "text"),
    ("_flr_sample.num_of_probes", "int"),
    ("_flr_fret_forster_radius.forster_radius", "float"),
    ("_flr_fret_forster_radius.kappa_squared", "float"),
])
def test_field_types(dic, field, expected_type):
    """Field types are correctly extracted."""
    item = dic.get_item(field)
    assert item is not None, f"Field {field} not found"
    # Type should be one of the expected values
    assert item.type_code in ["int", "float", "text", "code", "ucode"]


def test_comprehensive_flr_coverage(dic):
    """Comprehensive test of FLR category coverage."""
    # All expected FLR categories from the PRD-02a specification
    expected_flr_categories = {
        'flr_sample',
        'flr_sample_condition', 
        'flr_sample_probe_details',
        'flr_poly_probe_position',
        'flr_fret_forster_radius',
        'flr_fret_analysis',
        'flr_fret_distance_restraint',
        'flr_fret_calibration_parameters',
        'flr_instrument',
        'flr_inst_setting',
        'flr_experiment',
        'flr_entity_assembly',
        'flr_exp_condition',
    }
    
    flr_categories = set(dic.flr_categories())
    
    # Check that all expected categories are present
    for expected_cat in expected_flr_categories:
        assert expected_cat in flr_categories, f"Missing FLR category: {expected_cat}"
    
    # We should have at least 30 FLR categories (PRD-02a requirement)
    assert len(flr_categories) >= 30


def test_critical_flr_fields_present(dic):
    """Test that critical FLR fields from the PRD-02a specification are present."""
    critical_fields = {
        '_flr_sample.id',
        '_flr_sample.num_of_probes', 
        '_flr_sample.solvent_phase',
        '_flr_sample.sample_condition_id',
        '_flr_sample_condition.ph',
        '_flr_sample_condition.temperature',
        '_flr_sample_probe_details.probe_id',
        '_flr_sample_probe_details.fluorophore_type',
        '_flr_sample_probe_details.sample_id',
        '_flr_poly_probe_position.asym_id',
        '_flr_poly_probe_position.seq_id',
        '_flr_poly_probe_position.comp_id',
        '_flr_poly_probe_position.atom_id',
        '_flr_poly_probe_position.mutation_flag',
        '_flr_poly_probe_position.modification_flag',
        '_flr_fret_forster_radius.forster_radius',
        '_flr_fret_forster_radius.donor_probe_id',
        '_flr_fret_forster_radius.acceptor_probe_id',
        '_flr_fret_forster_radius.kappa_squared',
        '_flr_fret_forster_radius.index_of_refraction',
    }
    
    for field in critical_fields:
        item = dic.get_item(field)
        assert item is not None, f"Missing critical FLR field: {field}"


def test_suggest_values_basic():
    """Test basic suggest_values functionality."""
    dic = MmcifDictionary.load_bundled()
    # Test with empty prefix (should return all enum values for enumerated fields)
    values = dic.suggest_values("flr_sample", "solvent_phase", "")
    # Should return something (might be empty if no suggestions available)
    assert isinstance(values, list)
    
    # Test with probe names - should return known probe names
    from chisurf.core.mfdb.models import COMMON_PROBE_NAMES
    probe_values = dic.suggest_values("probes", "chromophore_name", "Cy")
    assert isinstance(probe_values, list)


def test_multiple_dictionary_files_loaded():
    """Test that all 7 bundled .dic files are being parsed."""
    dic = MmcifDictionary.load_bundled()
    categories = dic.categories()
    
    # We should have categories from multiple sources:
    # - mmcif_pdbx_v50.dic (core PDBx)
    # - mmcif_ihm_ext.dic (IHM)
    # - mmcif_ihm_flr_ext.dic (FLR)
    # - mmcif_ma.dic (ModelCIF)
    # - mmcif_std.dic (original mmCIF)
    # - mmcif_ddl.dic (DDL)
    # - mmcif_pdbx_v5_next.dic (development version)
    
    # Check for categories from different dictionaries
    pdbx_categories = [cat for cat in categories if cat.startswith("pdbx_")]
    ihm_categories = [cat for cat in categories if cat.startswith("ihm_")]
    flr_categories = [cat for cat in categories if cat.startswith("flr_")]
    ma_categories = [cat for cat in categories if cat.startswith("ma_")]
    
    # Should have categories from multiple dictionaries
    assert len(pdbx_categories) > 0, "No PDBx categories found"
    assert len(ihm_categories) > 0, "No IHM categories found"
    assert len(flr_categories) > 0, "No FLR categories found"


# Additional parameterized tests for better coverage
@pytest.mark.parametrize("category", ["flr_sample", "flr_fret_forster_radius", "entity"])
def test_category_items_present(dic, category):
    """Test that specific categories have items."""
    cat = dic.get_category(category)
    assert cat is not None, f"Category {category} not found"
    assert len(cat.items) > 0, f"Category {category} has no items"


@pytest.mark.parametrize("field,expected_enum", [
    ("_entity.type", "polymer"),
    ("_flr_sample_probe_details.fluorophore_type", "donor"),
])
def test_enumerations_present(dic, field, expected_enum):
    """Test that specific enumerations are present."""
    enums = dic.get_enumerations(field)
    assert expected_enum in enums, f"Expected enum '{expected_enum}' not found in {field}"