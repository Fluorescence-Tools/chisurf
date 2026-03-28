import sys
import os
import numpy as np
import json

# Add the parent directory to the path so we can import chinet
TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, TOPDIR)

import chinet as cn

def test_session_persist():
    print("\nTesting session persistence...")
    
    # Create a session with nodes and ports
    session = cn.Session()
    node1 = cn.Node("Node1")
    # We use constructor or property for value
    port1 = cn.Port()
    port1.value = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    port1.name = "Port1"
    node1.add_port("p1", port1, False)
    session.add_node("n1", node1)
    
    # Connect to "database" (memory) and write
    session.connect_to_db("memory", "memory", "memory", "memory")
    session.write_to_db()
    
    oid = session.oid
    print(f"Session OID: {oid}")
    
    # Save the session to a file
    filename = "test_session.chn"
    cn.save_session(filename)
    print(f"Session saved to {filename}")
    
    # Create a new session object and load from file
    cn.load_session(filename)
    print("Session loaded from file")
    
    session_reload = cn.Session()
    session_reload.connect_to_db("memory", "memory", "memory", "memory")
    success = session_reload.read_from_db(oid)
    
    assert success, "Failed to read session from reloaded database"
    assert len(session_reload.get_nodes()) == 1, f"Expected 1 node, got {len(session_reload.get_nodes())}"
    
    node_reload = session_reload.get_nodes()["n1"]
    port_reload = node_reload.get_port("p1")
    
    print(f"Reloaded port value: {port_reload}")
    # Use array view for comparison
    assert np.allclose(np.array(port_reload), np.array([1.0, 2.0, 3.0])), "Port value mismatch after reload"
    
    # Clean up
    if os.path.exists(filename):
        os.remove(filename)
    
    print("Session persistence test passed!")

def test_port_array_behavior():
    print("\nTesting Port array behavior...")
    
    p = cn.Port()
    p.value = np.array([10, 20, 30], dtype=np.int64)
    
    # Test numpy functions
    s = np.sum(p)
    print(f"Sum of port: {s}")
    assert s == 60, f"Expected sum 60, got {s}"
    
    # Test indexing
    assert p[1] == 20, f"Expected p[1]==20, got {p[1]}"
    
    # Test math
    p2 = np.array(p) * 2
    print(f"Port * 2: {p2}")
    assert np.all(p2 == [20, 40, 60]), "Math operation mismatch"
    
    # Test slicing
    subset = np.array(p)[1:]
    assert np.all(subset == [20, 30]), "Slicing mismatch"
    
    print("Port array behavior test passed!")

if __name__ == "__main__":
    test_session_persist()
    test_port_array_behavior()
    print("\nAll persistence tests completed!")
