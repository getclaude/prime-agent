"""Base class for MCP-client integrations exposed in the RLM kernel.

An integration is a Python skill package that subclasses :class:`McpIntegration`,
declares the MCP ``server`` it targets, and is imported in the kernel like any
other skill. Tools are auto-discovered from the server and bound as async
methods, so the agent writes ordinary Python:

    import linear
    issues = await linear.list_issues(team="Engineering")

Credentials live in the host's ``auth.json`` (single store, survives kernel
rebuilds). This module reads that file directly for the common case; on token
expiry it asks the host to refresh via ``rlm.host_request("mcp.refresh", ...)``
and re-reads. Interactive login runs host-side, never here.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from . import host_request

__all__ = ["McpDisabled", "McpIntegration", "McpToolError", "NotEnabled"]

# Stored access tokens are treated as expired this many seconds early so a token
# never dies mid-request. Mirrors the host's refresh buffer.
_EXPIRY_SKEW_SECONDS = 30


class NotEnabled(RuntimeError):
    """Raised when an integration has no usable credentials.

    The integration is installed but not logged in. The message tells the agent
    how to enable it so it can relay that to the user rather than retrying.
    """

    def __init__(self, server: str):
        self.server = server
        super().__init__(
            f"The '{server}' integration is not enabled: no credentials found. "
            f"Tell the user to run `/mcp login {server}` in Prime Agent to connect it. "
            f"Do not ask them to set environment variables."
        )


class McpDisabled(RuntimeError):
    """Raised when the host explicitly disables an integration in settings."""

    def __init__(self, server: str):
        self.server = server
        super().__init__(
            f"The '{server}' integration is disabled in settings. "
            f"Tell the user to enable it and run `/reload`."
        )


class McpToolError(RuntimeError):
    """Raised when an MCP tool call returns a result flagged as an error."""


def _agent_dir() -> Path:
    """Resolve the Prime Agent config dir the same way the rest of the runtime does."""
    raw = (
        os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")
        or os.environ.get("PI_CODING_AGENT_DIR")
        or str(Path.home() / ".prime" / "agent")
    )
    # resolve() so a relative env override reads auth.json from the right place,
    # not relative to the kernel's cwd.
    return Path(raw).expanduser().resolve()


def _read_auth(provider: str) -> dict[str, Any] | None:
    """Read one credential entry from auth.json. Returns None if absent/unreadable."""
    try:
        data = json.loads((_agent_dir() / "auth.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    cred = data.get(provider)
    return cred if isinstance(cred, dict) else None


def _resolve_config_value(value: str) -> str:
    """Resolve a stored api_key value the way the host does.

    A value may be a literal, an env-var name, or a `!command` indirection. The
    command form can't run safely in the kernel (the host injects those resolved),
    so skip it; otherwise treat the value as an env-var name if set, else literal.
    """
    value = value.strip()
    if not value or value.startswith("!"):
        return ""
    return (os.environ.get(value) or value).strip()


def _resolve_streamable_http():
    """Return an SDK streamable-HTTP transport callable.

    SDK versions vary: some expose ``streamablehttp_client(url, headers=...)``,
    others ``streamable_http_client(url, *, http_client=...)``, and some expose
    both with *different* signatures. Imported lazily so importing an integration
    package never hard-fails when ``mcp`` is absent.
    """
    from mcp.client import streamable_http as mod  # noqa: PLC0415

    for name in ("streamablehttp_client", "streamable_http_client"):
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn
    raise ImportError(
        "the installed `mcp` SDK exposes no streamable-HTTP client; upgrade `mcp`"
    )


def _build_mcp_http_client(headers: dict[str, str]):
    """Return an SDK-configured HTTP client so MCP timeout defaults are retained."""
    from mcp.shared._httpx_utils import create_mcp_http_client  # noqa: PLC0415

    return create_mcp_http_client(headers=headers)


class McpIntegration:
    """Subclass and set :attr:`server` (and :attr:`url` for remote servers).

    Tools are discovered on first use and bound as async methods via
    ``__getattr__``; ``await self.call_tool(name, args)`` is the explicit escape
    hatch and the hook for hand-written typed wrappers.
    """

    #: Credential / config key for this integration (matches the auth.json entry
    #: ``mcp:<server>`` and the mcpServers settings key).
    server: str = ""

    #: Remote MCP endpoint. Required unless a subclass overrides ``_open_streams``.
    url: str | None = None

    #: Optional env var holding a static bearer token (used instead of auth.json OAuth).
    bearer_token_env: str | None = None

    def __init__(self, server: str | None = None) -> None:
        if server is not None:
            self.server = server.strip()
        if not self.server:
            raise ValueError(f"{type(self).__name__} must set a non-empty `server`")
        self._tools: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._enabled_tools: set[str] | None = None
        self._disabled_tools: set[str] = set()

    # -- credentials --------------------------------------------------------

    @property
    def _provider_id(self) -> str:
        return f"mcp:{self.server}"

    def _token(self, bearer_token_env: str | None = None) -> str | None:
        """Current usable bearer token, or None if missing/expired (needs refresh).

        A static bearer-token env var wins (matches the host's `isAuthed` check);
        otherwise read auth.json. OAuth tokens are only returned while still fresh.
        """
        token_env = bearer_token_env or self.bearer_token_env
        if token_env:
            env_token = os.environ.get(token_env, "").strip()
            if env_token:
                return env_token
        cred = _read_auth(self._provider_id)
        if cred is None:
            return None
        if cred.get("type") == "api_key":
            return _resolve_config_value(str(cred.get("key") or "")) or None
        # OAuth credential: {access, refresh, expires(ms)}.
        access = str(cred.get("access") or "")
        expires = cred.get("expires")
        fresh = isinstance(expires, (int, float)) and (
            time.time() * 1000 < expires - _EXPIRY_SKEW_SECONDS * 1000
        )
        if access and fresh:
            return access
        return None  # signal: needs refresh

    async def _resolve_token(self, bearer_token_env: str | None = None) -> str:
        token = self._token(bearer_token_env)
        if token:
            return token
        # Expired or missing-access: ask the host to refresh, then re-validate via
        # _token() (which re-checks expiry) rather than trusting any access value.
        if _read_auth(self._provider_id) is not None:
            refresh_error: Exception | None = None
            try:
                await host_request("mcp.refresh", {"server": self.server})
            except RuntimeError as exc:
                refresh_error = exc
            token = self._token(bearer_token_env)
            if token:
                return token
            # A refresh that failed (vs. genuinely-absent creds) is a recoverable
            # error; don't mislabel it as "not enabled / re-login".
            if refresh_error is not None:
                raise RuntimeError(
                    f"Failed to refresh credentials for '{self.server}': {refresh_error}"
                ) from refresh_error
        raise NotEnabled(self.server)

    # -- connection ---------------------------------------------------------

    async def _resolve_transport_config(self) -> dict[str, Any]:
        """Resolve and validate the transport declaration supplied by the host."""
        try:
            cfg = await host_request("mcp.config", {"server": self.server})
        except RuntimeError:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        if cfg.get("enabled") is False:
            raise McpDisabled(self.server)

        transport_type = cfg.get("type")
        if transport_type == "stdio" or (transport_type is None and "command" in cfg):
            command = cfg.get("command")
            args = cfg.get("args", [])
            env = cfg.get("env")
            cwd = cfg.get("cwd")
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"MCP server '{self.server}' has no stdio command")
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise TypeError(f"MCP server '{self.server}' stdio args must be strings")
            if env is not None and (
                not isinstance(env, dict)
                or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items())
            ):
                raise TypeError(f"MCP server '{self.server}' stdio env must contain string values")
            if cwd is not None and not isinstance(cwd, str):
                raise TypeError(f"MCP server '{self.server}' stdio cwd must be a string")
            return {
                "type": "stdio",
                "command": command,
                "args": args,
                "env": env,
                "cwd": cwd,
                "enabledTools": cfg.get("enabledTools"),
                "disabledTools": cfg.get("disabledTools"),
            }

        url = cfg.get("url")
        if not (isinstance(url, str) and url):
            url = self.url
        headers = cfg.get("headers")
        extra = headers if isinstance(headers, dict) else {}
        return {
            "type": "http",
            "url": url,
            "headers": {str(key): str(value) for key, value in extra.items()},
            "requiresAuth": cfg.get("requiresAuth", True),
            "allowStoredCredentials": cfg.get("allowStoredCredentials", True),
            "bearerTokenEnvVar": cfg.get("bearerTokenEnvVar"),
            "enabledTools": cfg.get("enabledTools"),
            "disabledTools": cfg.get("disabledTools"),
        }

    async def _resolve_config(self) -> tuple[str | None, dict[str, str]]:
        """Backward-compatible HTTP URL and headers resolver for integrations."""
        cfg = await self._resolve_transport_config()
        return cfg.get("url"), cfg.get("headers", {})

    def _set_tool_filters(self, cfg: dict[str, Any]) -> None:
        enabled = cfg.get("enabledTools")
        disabled = cfg.get("disabledTools")
        if enabled is not None and (
            not isinstance(enabled, list) or not all(isinstance(name, str) for name in enabled)
        ):
            raise TypeError(f"MCP server '{self.server}' enabledTools must contain strings")
        if disabled is not None and (
            not isinstance(disabled, list) or not all(isinstance(name, str) for name in disabled)
        ):
            raise TypeError(f"MCP server '{self.server}' disabledTools must contain strings")
        self._enabled_tools = set(enabled) if enabled is not None else None
        self._disabled_tools = set(disabled or [])

    def _tool_allowed(self, tool: str) -> bool:
        return (
            (self._enabled_tools is None or tool in self._enabled_tools)
            and tool not in self._disabled_tools
        )

    async def _open_session(self, stack: AsyncExitStack):
        """Open an initialized MCP ClientSession bound to ``stack``.

        The host declaration selects streamable HTTP or stdio. Both transports
        use the official MCP Python SDK and are closed with ``stack``.
        """
        import inspect  # noqa: PLC0415

        cfg = await self._resolve_transport_config()
        self._set_tool_filters(cfg)
        from mcp import ClientSession  # noqa: PLC0415

        if cfg["type"] == "stdio":
            from mcp.client.stdio import (  # noqa: PLC0415
                StdioServerParameters,
                stdio_client,
            )

            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg["args"],
                env=cfg["env"],
                cwd=cfg["cwd"],
            )
            read, write, *_ = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return session

        url = cfg.get("url")
        if not url:
            raise ValueError(
                f"MCP server '{self.server}' has neither a stdio command nor an HTTP URL"
            )
        extra_headers = cfg["headers"]
        if cfg.get("requiresAuth") is True:
            token_env = cfg.get("bearerTokenEnvVar")
            if isinstance(token_env, str):
                token = os.environ.get(token_env, "").strip()
                if not token:
                    raise NotEnabled(self.server)
            elif cfg.get("allowStoredCredentials") is True:
                token = await self._resolve_token()
            else:
                raise NotEnabled(self.server)
            extra_headers = {**extra_headers, "Authorization": f"Bearer {token}"}
        transport = _resolve_streamable_http()

        # SDK signatures vary: some take headers=, others only http_client=.
        params = inspect.signature(transport).parameters
        if "headers" in params:
            cm = transport(url, headers=extra_headers)
        elif "http_client" in params:
            client = await stack.enter_async_context(_build_mcp_http_client(extra_headers))
            cm = transport(url, http_client=client)
        else:
            raise RuntimeError(
                f"unsupported mcp streamable-HTTP client signature: {tuple(params)}"
            )

        read, write, *_ = await stack.enter_async_context(cm)
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    # -- tools --------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tools as ``[{name, description, inputSchema}]``."""
        await self._ensure_tools()
        return [dict(t) for t in (self._tools or {}).values()]

    async def _ensure_tools(self) -> None:
        if self._tools is not None:
            return
        async with self._lock:
            if self._tools is not None:
                return
            async with AsyncExitStack() as stack:
                session = await self._open_session(stack)
                resp = await session.list_tools()
                self._tools = {
                    t.name: {
                        "name": t.name,
                        "description": getattr(t, "description", "") or "",
                        "inputSchema": (
                            getattr(t, "inputSchema", None)
                            or getattr(t, "input_schema", None)
                            or {}
                        ),
                    }
                    for t in resp.tools
                    if self._tool_allowed(t.name)
                }

    async def call_tool(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call ``tool`` on the server and return its parsed result.

        Opens a fresh session per call: MCP sessions are not safe to hold across
        the kernel's snapshot/restore, and per-call connect keeps this robust to
        idle sessions and token rotation at modest latency cost.
        """
        async with AsyncExitStack() as stack:
            session = await self._open_session(stack)
            if not self._tool_allowed(tool):
                raise PermissionError(
                    f"MCP tool '{self.server}.{tool}' is disabled by enabledTools/disabledTools"
                )
            result = await session.call_tool(tool, arguments or {})
        return _parse_result(result)

    def __getattr__(self, name: str):
        # Only reached for names not found normally; bind as an async tool call.
        if name.startswith("_"):
            raise AttributeError(name)

        async def _call(**kwargs: Any) -> Any:
            await self._ensure_tools()
            if self._tools is not None and name not in self._tools:
                available = ", ".join(sorted(self._tools)) or "(none)"
                raise AttributeError(
                    f"'{self.server}' has no tool '{name}'. Available: {available}"
                )
            return await self.call_tool(name, kwargs)

        _call.__name__ = name
        _call.__qualname__ = f"{type(self).__name__}.{name}"
        if self._tools and name in self._tools:
            schema = self._tools[name].get("inputSchema") or {}
            desc = self._tools[name].get("description") or ""
            _call.__doc__ = f"{desc}\n\nArguments (JSON Schema):\n{json.dumps(schema, indent=2)}"
        return _call


def _parse_result(result: Any) -> Any:
    """Normalize a CallToolResult into plain Python (structured output preferred).

    Raises McpToolError when the server flags the result as an error, so a failed
    tool call doesn't look like a successful one to the caller.
    """
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    if getattr(result, "isError", getattr(result, "is_error", False)):
        raise McpToolError("\n".join(texts) or "MCP tool returned an error")

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:  # falsy-but-valid payloads ({} / []) are real results
        return structured
    if texts:
        return "\n".join(texts)

    # Non-text content (images, embedded resources): return them as plain dicts
    # rather than the opaque SDK object so callers get usable data.
    blocks = getattr(result, "content", None) or []
    if blocks:
        return [b.model_dump(mode="json") if hasattr(b, "model_dump") else b for b in blocks]
    return result
