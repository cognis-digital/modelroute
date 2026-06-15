"""MODELROUTE MCP server — exposes route resolution as an MCP tool."""
from __future__ import annotations

import json


def serve() -> int:
    """Start an MCP stdio server. Requires the optional 'mcp' extra:
        pip install "cognis-modelroute[mcp]"
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("Install the MCP extra: pip install 'cognis-modelroute[mcp]'")
        return 1

    from modelroute.core import RouteError, resolve

    app = FastMCP("modelroute")

    @app.tool()
    def modelroute_resolve(alias: str, strategy: str = "local-first") -> str:
        """Resolve a model alias to an ordered fallback chain.

        Returns JSON with the candidate list or an error message.
        """
        try:
            chain = resolve(alias, strategy=strategy)
            return json.dumps([c.to_dict() for c in chain], indent=2)
        except RouteError as exc:
            return json.dumps({"error": str(exc)})

    app.run()
    return 0
