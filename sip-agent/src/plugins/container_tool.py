"""
Container Control Tool Plugin
=============================
Check status of, or restart, docker containers on the host — restricted to an
explicit allowlist (CONTAINER_CTL_ALLOWLIST) and requiring the docker socket to
be mounted into the agent container (DOCKER_SOCKET_PATH).

This is a DANGEROUS tool: the requested container name is validated against the
allowlist before any I/O, and restarts additionally require confirm=true.

Usage in conversation:
User: "Is the n8n container up?"
LLM: [TOOL:CONTAINER_CTL:action=status,name=n8n]

User: "Yes, restart it."
LLM: [TOOL:CONTAINER_CTL:action=restart,name=n8n,confirm=true]
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)


def _allowlist(config) -> List[str]:
    """Parse the comma-separated container allowlist from config."""
    if not config or not config.container_ctl_allowlist:
        return []
    return [
        name.strip()
        for name in config.container_ctl_allowlist.split(",")
        if name.strip()
    ]


async def _docker_request(
    socket_path: str,
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Any]:
    """Issue a Docker Engine API request over the unix socket.

    Returns (status_code, parsed_json_or_None). Never raises: any transport
    or parse failure returns (0, None).
    """
    try:
        transport = httpx.AsyncHTTPTransport(uds=socket_path)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://docker", timeout=15.0
        ) as client:
            response = await client.request(method, path, params=params)
            try:
                body = response.json()
            except Exception:
                body = None
            return response.status_code, body
    except Exception as e:
        logger.error(f"Docker API request failed ({method} {path}): {e}")
        return 0, None


class ContainerControlTool(BaseTool):
    """Restart or inspect allowlisted docker containers via the docker socket."""

    name = "CONTAINER_CTL"
    description = (
        "Check status of, or restart, an allowlisted docker container. "
        "Always ask the caller to confirm before restarting, then call again "
        "with confirm=true."
    )
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "action": {
            "type": "string",
            "description": "What to do: 'status' or 'restart'",
            "required": False,
            "default": "status",
        },
        "name": {
            "type": "string",
            "description": "The container name to act on",
            "required": True,
        },
        "confirm": {
            "type": "boolean",
            "description": "Must be true to actually restart a container",
            "required": False,
            "default": False,
        },
    }

    def __init__(self, assistant):
        super().__init__(assistant)
        if not self.config:
            self.enabled = False
            logger.info("CONTAINER_CTL tool disabled - no config available")
        elif not _allowlist(self.config):
            self.enabled = False
            logger.info("CONTAINER_CTL tool disabled - allowlist is empty")
        elif not os.path.exists(self.config.docker_socket_path):
            self.enabled = False
            logger.info(
                "CONTAINER_CTL tool disabled - docker socket not found at "
                f"{self.config.docker_socket_path}"
            )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        action = (params.get("action") or "status").lower()
        name = (params.get("name") or "").strip()
        confirm = bool(params.get("confirm", False))

        # Security gate: exact, case-sensitive allowlist match before ANY I/O.
        # Never reveal the allowlist contents in the spoken message.
        if name not in _allowlist(self.config):
            log_event(
                logger, logging.WARNING,
                f"CONTAINER_CTL denied for container '{name}'",
                event="container_ctl_denied",
            )
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I am not allowed to touch that container.",
                data={"container": name, "action": action},
            )

        if action == "status":
            return await self._status(name)
        if action == "restart":
            return await self._restart(name, confirm)

        return ToolResult(
            status=ToolStatus.FAILED,
            message=f"I don't know how to {action} a container. I can check status or restart.",
            data={"container": name, "action": action},
        )

    async def _status(self, name: str) -> ToolResult:
        code, data = await _docker_request(
            self.config.docker_socket_path, "GET", f"/containers/{name}/json"
        )

        if code == 0:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I cannot reach the docker daemon.",
                data={"container": name, "action": "status"},
            )
        if code == 404:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"I do not see a container called {name}.",
                data={"container": name, "action": "status"},
            )
        if code != 200 or not isinstance(data, dict):
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"I could not get the status of {name}.",
                data={"container": name, "action": "status"},
            )

        state = data.get("State") or {}
        status = state.get("Status") or "unknown"
        health = (state.get("Health") or {}).get("Status")

        message = f"The {name} container is {status}."
        if health:
            message += f" Health is {health}."

        log_event(
            logger, logging.INFO,
            f"CONTAINER_CTL status for {name}: {status}",
            event="container_ctl_status",
        )
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"container": name, "action": "status", "status": status,
                  "health": health},
        )

    async def _restart(self, name: str, confirm: bool) -> ToolResult:
        if not confirm:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Tell me to confirm and I will restart {name}.",
                data={"container": name, "action": "restart", "confirmed": False},
            )

        code, _ = await _docker_request(
            self.config.docker_socket_path, "POST",
            f"/containers/{name}/restart", params={"t": 10},
        )

        if code == 0:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I cannot reach the docker daemon.",
                data={"container": name, "action": "restart", "confirmed": True},
            )
        if code == 404:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"I do not see a container called {name}.",
                data={"container": name, "action": "restart", "confirmed": True},
            )
        if code == 204:
            log_event(
                logger, logging.INFO,
                f"CONTAINER_CTL restarted {name}",
                event="container_ctl_restart",
            )
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"Restarting {name} now.",
                data={"container": name, "action": "restart", "confirmed": True},
            )

        return ToolResult(
            status=ToolStatus.FAILED,
            message=f"I could not restart {name}.",
            data={"container": name, "action": "restart", "confirmed": True},
        )
