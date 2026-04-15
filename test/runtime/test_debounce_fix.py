import unittest
import time
import threading
from chisurf.runtime.actions import ActionSpec, ActionRegistry, ActionDispatcher

class TestDebounceFix(unittest.TestCase):
    def test_trailing_edge_debounce(self):
        """Verify that the final call in a debounced sequence is eventually executed."""
        reg = ActionRegistry()
        calls = []
        
        def handler(value):
            calls.append(value)
            
        reg.register(ActionSpec(
            "test.action", 
            schema={"value": int}, 
            handler=handler, 
            debounce_ms=100, 
            debounce_keys=("identity",)
        ))
        
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: None)
        
        # 1. First call (leading edge) - should fire immediately
        dispatcher.execute("test.action", {"value": 1, "identity": "A"})
        self.assertEqual(calls, [1])
        
        # 2. Rapid second call - should be debounced but scheduled
        dispatcher.execute("test.action", {"value": 2, "identity": "A"})
        self.assertEqual(calls, [1]) # Still only the first one
        
        # 3. Rapid third call - should replace the second one as the trailing edge
        dispatcher.execute("test.action", {"value": 3, "identity": "A"})
        self.assertEqual(calls, [1])
        
        # 4. Wait for debounce to expire
        time.sleep(0.15)
        
        # Now it should have fired the last one
        self.assertEqual(calls, [1, 3])
        
    def test_independent_debounce_streams(self):
        """Verify that different identities are debounced independently."""
        reg = ActionRegistry()
        calls = []
        
        def handler(identity, value):
            calls.append((identity, value))
            
        reg.register(ActionSpec(
            "test.action", 
            schema={"identity": str, "value": int}, 
            handler=handler, 
            debounce_ms=100, 
            debounce_keys=("identity",)
        ))
        
        dispatcher = ActionDispatcher(registry=reg, history_provider=lambda: None)
        
        # Fire identity A
        dispatcher.execute("test.action", {"identity": "A", "value": 1})
        # Fire identity B immediately - should NOT be debounced by A
        dispatcher.execute("test.action", {"identity": "B", "value": 10})
        
        self.assertEqual(len(calls), 2)
        self.assertIn(("A", 1), calls)
        self.assertIn(("B", 10), calls)

if __name__ == "__main__":
    unittest.main()
