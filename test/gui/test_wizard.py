import pytest
import chisurf.core.models.tcspc.lifetime

def test_lifetime_append_pop_sync():
    """
    Verify that popping and appending components updates the 
    internal parameter cache and component count (n) correctly
    to avoid duplicate parameter names (like xL2 appearing twice).
    """
    lt = chisurf.core.models.tcspc.lifetime.Lifetime(short='L')
    assert lt.n == 0, "Initial n should be 0"
    
    # Append first component
    lt.append()
    assert lt.n == 1, "n should be 1 after first append"
    assert 'xL1' in lt.parameters_all_dict
    assert 'tL1' in lt.parameters_all_dict
    
    # Pop the component
    lt.pop()
    assert lt.n == 0, "n should return to 0 after pop"
    assert 'xL1' not in lt.parameters_all_dict
    assert 'tL1' not in lt.parameters_all_dict
    
    # Append again (should be component 1, not 2)
    lt.append()
    assert lt.n == 1, "n should be 1 after appending again"
    assert 'xL1' in lt.parameters_all_dict
    
    # Append another (should be component 2)
    lt.append()
    assert lt.n == 2, "n should be 2 after second append"
    assert 'xL2' in lt.parameters_all_dict
    assert 'xL3' not in lt.parameters_all_dict

def test_init_chisurf_onboarding_wizard_import():
    """
    Ensure the onboarding wizard plugin can be imported without errors.
    """
    try:
        from chisurf.plugins._dev.init_chisurf import wizard as _wizard  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Could not import onboarding wizard: {e}")
