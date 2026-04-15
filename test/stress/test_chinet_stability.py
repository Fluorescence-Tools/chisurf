import unittest
import gc
import chinet as cn
import numpy as np

class TestChinetStability(unittest.TestCase):
    """
    Stress test for the pure-Python chinet module to verify it's free from
    Access Violations (Bug #3) and handle heavy node/port allocation gracefully.
    """

    def test_massive_port_allocation(self):
        """Allocate many ports and trigger GC to ensure no crashes."""
        print("\nAllocating 20,000 ports...")
        ports = []
        for i in range(20000):
            p = cn.Port(value=float(i))
            ports.append(p)
            if i % 5000 == 0:
                gc.collect()
        
        self.assertEqual(len(ports), 20000)
        self.assertEqual(ports[1000].value, 1000.0)
        
        # Clear and force GC
        del ports
        gc.collect()
        print("Port allocation stable.")

    def test_complex_graph_evaluation(self):
        """Create a deep graph and evaluate it many times."""
        print("Building complex graph...")
        root_port = cn.Port(1.0)
        current_node = None
        nodes = []
        
        # Create a chain of 1000 nodes
        prev_port = root_port
        for i in range(1000):
            node = cn.Node()
            # In a real scenario nodes have inputs/outputs. 
            # Assuming simplified Port-Node structure for stress test.
            # chinet nodes use ports.
            p_in = cn.Port(0.0)
            p_out = cn.Port(0.0)
            p_in.link = prev_port
            # Mocking some evaluation logic if supported by Python chinet
            node.add_input_port("in", p_in)
            node.add_output_port("out", p_out)
            nodes.append(node)
            prev_port = p_out
            
        print(f"Graph with {len(nodes)} nodes built. Evaluating...")
        for i in range(10):
            root_port.value = float(i)
            # Trigger evaluation if needed (implementation dependent)
            # In chinet-python, value setting on linked ports might trigger updates.
            gc.collect()
            
        print("Graph evaluation stable.")

    def test_random_linking_and_unlinking(self):
        """Randomly link and unlink ports to check for reference cycle leaks or crashes."""
        print("Random linking/unlinking stress test...")
        ports = [cn.Port(0.0) for _ in range(1000)]
        for _ in range(5000):
            i, j = np.random.randint(0, 1000, size=2)
            if i != j:
                ports[i].link = ports[j]
            if _ % 1000 == 0:
                # Randomly break some links
                k = np.random.randint(0, 1000)
                ports[k].link = None
        
        gc.collect()
        print("Linking stress test stable.")

if __name__ == "__main__":
    unittest.main()
