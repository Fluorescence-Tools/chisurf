import os
import shutil
import tempfile
import unittest
import numpy as np
import chinet
from chisurf.project.project import Project

class TestChinetSession(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Ensure a clean session for every test if possible
        try:
            chinet.session.nodes.clear()
        except (AttributeError, TypeError):
            pass

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        try:
            chinet.session.nodes.clear()
        except (AttributeError, TypeError):
            pass

    def test_basic_session_serialization(self):
        # 1. Create a simple network
        # Node A (Port 1)
        node_a = chinet.Node()
        node_a.name = "NodeA"
        port_a = chinet.Port(10.0)
        port_a.name = "out"
        node_a.add_output_port("out", port_a)
        
        # Node B (Input 1)
        node_b = chinet.Node()
        node_b.name = "NodeB"
        port_b = chinet.Port(0.0)
        port_b.name = "in"
        node_b.add_input_port("in", port_b)
        
        # Connect them
        port_b.link(port_a)
        
        # Verify initial state
        self.assertEqual(node_b.inputs["in"].value, 10.0)
        
        # 2. Save via Project
        proj = Project(name="ChinetTest")
        proj_dir = os.path.join(self.test_dir, "proj1")
        proj.save(proj_dir)
        
        session_file = os.path.join(proj_dir, "session.jsonl")
        self.assertTrue(os.path.exists(session_file))
        
        # 3. Modify current session or clear it
        chinet.session.clear()
        # Verify it's gone from local memory (standard chinet behavior might vary depending on how it's wrapped)
        # For this test, we rely on Project.load to restore it.
        
        # 4. Load via Project
        proj_loaded = Project.load(proj_dir)
        self.assertEqual(proj_loaded.name, "ChinetTest")
        
        # 5. Verify restored session
        # We need to find the nodes in the restored session. 
        # Note: chinet.session.nodes often returns a list/dict of current session nodes.
        nodes = chinet.session.nodes
        self.assertIn("NodeA", nodes)
        self.assertIn("NodeB", nodes)
        
        restored_node_a = nodes["NodeA"]
        restored_node_b = nodes["NodeB"]
        
        self.assertEqual(restored_node_a.outputs["out"].value, 10.0)
        self.assertEqual(restored_node_b.inputs["in"].value, 10.0)
        
        # Verify connection persists
        restored_node_a.outputs["out"].value = 25.0
        # Some chinet versions require explicit evaluation or reactiveness
        # restored_node_b.evaluate() # If not reactive
        self.assertEqual(restored_node_b.inputs["in"].value, 25.0)

if __name__ == "__main__":
    unittest.main()
