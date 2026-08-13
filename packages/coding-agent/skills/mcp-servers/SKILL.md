---
name: mcp-servers
description: Use MCP servers configured in Prime Agent settings without creating a Python skill per server. Supports local stdio and remote Streamable HTTP transports.
---

# Configured MCP servers

Use the generic `mcp_servers` module from IPython for servers declared under
`mcpServers` in `~/.prime/agent/settings.json`.

```python
import mcp_servers

await mcp_servers.list_servers()
server = mcp_servers.get("codegraph")
await server.list_tools()
result = await server.call_tool("search", {"query": "McpManager"})
```

For server names that are valid Python identifiers, attribute access is also
available: `await mcp_servers.codegraph.list_tools()`.

Always discover tools before calling them. Tool names and input schemas are
defined by the server. Every operation is async. `enabledTools` is an allowlist;
`disabledTools` always denies matching tools.

Local stdio servers execute their configured command with the Prime Agent
process user's permissions. Only global stdio declarations are executable;
project-local stdio declarations remain blocked until project MCP trust is
implemented.
