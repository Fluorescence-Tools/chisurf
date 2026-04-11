import unittest
from chisurf.mcp.server import create_mcp

class TestMcpServer(unittest.TestCase):

    def test_mcp_registration(self):
        """Verify that the MCP server can be created and tools are registered."""
        # Note: We don't mcp.run() because it's blocking (stdio/sse)
        mcp = create_mcp(name="TestChiSurf")
        
        self.assertEqual(mcp.name, "TestChiSurf")
        
        # Verify core tools are registered
        # fastmcp stores tools in mcp.tools
        tool_names = [t.name for t in mcp.tools]
        
        self.assertIn("ping", tool_names)
        self.assertIn("list_runtime_vars", tool_names)
        self.assertIn("get_runtime_var", tool_names)
        self.assertIn("describe_state", tool_names)
        self.assertIn("run_script", tool_names)

    def test_ping_tool(self):
        """Test the ping tool specifically if possible without a running server."""
        mcp = create_mcp()
        ping_tool = next(t for t in mcp.tools if t.name == "ping")
        
        # We can call the underlying function directly
        result = ping_tool.func()
        self.assertTrue(result["ok"])
        self.assertEqual(result["server"], "chisurf.mcp")

if __name__ == "__main__":
    unittest.main()
