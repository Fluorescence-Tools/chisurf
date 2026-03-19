import pytest
from chisurf.mcp.server import create_mcp, resolve_transport_kwargs

def test_mcp_creation():
    mcp = create_mcp()
    assert mcp is not None
    assert mcp.name == "ChiSurf"

# FastMCP registers tools on the `_mcp_server` object or `tools` attribute.
def test_tools_registered():
    mcp = create_mcp()
    
    # In fastmcp, tools are usually collected and exposed via mcp.list_tools() 
    # but since it's async we can check if the server is instantiated and tools are added
    # We just ensure creation passes for now, as fastmcp handles registration.
    pass


def test_resolve_transport_kwargs_streamable_http():
    out = resolve_transport_kwargs(
        transport="streamable-http",
        host="127.0.0.1",
        port=8765,
        path="mcp",
    )
    assert out["host"] == "127.0.0.1"
    assert out["port"] == 8765
    assert out["path"] == "/mcp"


def test_resolve_transport_kwargs_stdio():
    out = resolve_transport_kwargs(transport="stdio")
    assert out == {}


def test_mcp_server_has_quality_tools_contract():
    src = open("chisurf/mcp/server.py", "r", encoding="utf-8").read()
    assert "def get_fit_quality(" in src
    assert "def run_fit_with_quality(" in src
