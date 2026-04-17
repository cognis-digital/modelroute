"""MODELROUTE MCP server — exposes scan() as an MCP tool for Cognis.Studio."""
from __future__ import annotations
from modelroute.core import scan, to_json

def serve() -> int:
    """Start an MCP stdio server. Requires the optional 'mcp' extra:
        pip install "cognis-modelroute[mcp]"
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        print("Install the MCP extra: pip install 'cognis-modelroute[mcp]'")
        return 1
    app = FastMCP("modelroute")

    @app.tool()
    def modelroute_scan(target: str) -> str:
        """Local model router / proxy across Ollama, vLLM, and cloud with fallback. Returns JSON findings."""
        return to_json(scan(target))

    app.run()
    return 0
