import unittest
import chisurf.core.models.tcspc.lifetime
from chisurf.core.fitting.parameter import FittingParameter

class TestLifetimePopAppendBug(unittest.TestCase):
    def test_lifetime_append_pop_sync(self):
        """
        Verify that popping and appending components updates the 
        internal parameter cache and component count (n) correctly
        to avoid duplicate parameter names (like xL2 appearing twice).
        """
        lt = chisurf.core.models.tcspc.lifetime.Lifetime(short='L')
        self.assertEqual(lt.n, 0, "Initial n should be 0")
        
        # Append first component
        lt.append()
        self.assertEqual(lt.n, 1, "n should be 1 after first append")
        self.assertIn('xL1', lt.parameters_all_dict)
        self.assertIn('tL1', lt.parameters_all_dict)
        
        # Pop the component
        lt.pop()
        self.assertEqual(lt.n, 0, "n should return to 0 after pop")
        self.assertNotIn('xL1', lt.parameters_all_dict)
        self.assertNotIn('tL1', lt.parameters_all_dict)
        
        # Append again (should be component 1, not 2)
        lt.append()
        self.assertEqual(lt.n, 1, "n should be 1 after appending again")
        self.assertIn('xL1', lt.parameters_all_dict)
        
        # Append another (should be component 2)
        lt.append()
        self.assertEqual(lt.n, 2, "n should be 2 after second append")
        self.assertIn('xL2', lt.parameters_all_dict)
        self.assertNotIn('xL3', lt.parameters_all_dict)

if __name__ == '__main__':
    unittest.main()
