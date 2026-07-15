"""Tiny MCP fixture server for the component round-trip tests.

Launched by the tests over the stdio transport (command=sys.executable,
args=[this file]). Exposes a few deliberately simple tools:

- echo(text)      -> "echo: <text>"
- add(a, b)       -> a + b
- secret()        -> must NEVER be exposed (tests the expose allowlist)
- danger()        -> gated behind confirm=true in the test servers file
- slow(seconds)   -> sleeps; used to test MCP_TOOL_TIMEOUT_S enforcement
- tally(options)  -> nested/object param (tests the JSON-string round trip)
"""
from typing import Dict

import anyio

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fixture")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back."""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def secret() -> str:
    """Not on the expose allowlist — must never register."""
    return "should never be exposed"


@mcp.tool()
def danger() -> str:
    """A destructive action gated behind confirm=true."""
    return "danger done"


@mcp.tool()
def tally(options: Dict[str, int]) -> str:
    """Sum the integer values of a JSON object (nested/object param)."""
    return f"tally: {sum(options.values())}"


@mcp.tool()
async def slow(seconds: float = 30.0) -> str:
    """Sleep for a while (used to trigger the tool-call timeout)."""
    await anyio.sleep(seconds)
    return "finally done"


if __name__ == "__main__":
    mcp.run("stdio")
