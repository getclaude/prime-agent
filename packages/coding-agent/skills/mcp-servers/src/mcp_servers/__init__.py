"""Generic access to MCP servers configured in Prime Agent settings."""

from __future__ import annotations

from typing import Any

from rlm import McpIntegration, host_request

__all__ = ["call_tool", "get", "list_servers", "list_tools"]

_instances: dict[str, McpIntegration] = {}
_RESERVED = {"run", "__wrapped__", "__call__"}


async def list_servers() -> list[dict[str, str]]:
    """Return enabled user-configured MCP server names and transport types."""
    payload = await host_request("mcp.list_configured")
    servers = payload.get("servers")
    if not isinstance(servers, list):
        raise RuntimeError("mcp.list_configured returned an invalid server list")
    return [dict(server) for server in servers if isinstance(server, dict)]


def get(server: str) -> McpIntegration:
    """Return a cached dynamic integration for one configured server."""
    if not isinstance(server, str) or not server.strip():
        raise ValueError("server must be a non-empty string")
    name = server.strip()
    integration = _instances.get(name)
    if integration is None:
        integration = McpIntegration(name)
        _instances[name] = integration
    return integration


async def list_tools(server: str) -> list[dict[str, Any]]:
    """Discover tools exposed by a configured server."""
    return await get(server).list_tools()


async def call_tool(
    server: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Call a tool exposed by a configured server."""
    return await get(server).call_tool(tool, arguments)


def __getattr__(name: str) -> McpIntegration:
    if name.startswith("_") or name in _RESERVED:
        raise AttributeError(name)
    return get(name)
