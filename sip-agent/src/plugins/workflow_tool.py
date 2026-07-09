"""
Workflow Trigger Tool Plugin
============================
Fires a pre-registered webhook (n8n or any HTTP endpoint) by name, turning
every workflow into a voice command. Webhooks are registered in a JSON file
at ``<data_dir>/workflows.json`` which is re-read on every invocation, so it
can be live-edited without restarting the agent:

    {
      "lights_off": {
        "url": "http://n8n:5678/webhook/abc",
        "description": "Turn off the living room lights"
      }
    }

Usage in conversation:
User: "Turn off the lights"
LLM: [TOOL:TRIGGER_WORKFLOW:name=lights_off]
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "workflows.json"


def _load_registry(path: Path) -> Dict[str, Dict[str, str]]:
    """Load the workflow registry, returning {} on any problem.

    Malformed entries (non-dict values or entries without a url) are dropped
    so one bad line in a live-edited file can't take out the whole registry.
    """
    try:
        if not path.exists():
            logger.warning(f"Workflow registry not found: {path}")
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not read workflow registry {path}: {e}")
        return {}

    if not isinstance(raw, dict):
        logger.warning(f"Workflow registry {path} is not a JSON object; ignoring")
        return {}

    registry: Dict[str, Dict[str, str]] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict) and entry.get("url"):
            registry[str(name)] = entry
        else:
            logger.warning(f"Skipping malformed workflow entry: {name}")
    return registry


def _speakable(name: str) -> str:
    """Make a registry key pleasant to read aloud (lights_off -> lights off)."""
    return name.replace("_", " ").replace("-", " ").strip()


class TriggerWorkflowTool(BaseTool):
    """Trigger a named, pre-registered automation webhook."""

    name = "TRIGGER_WORKFLOW"
    description = ("Run a named automation workflow by firing its registered "
                   "webhook, for example an n8n workflow")
    enabled = True
    speak_result = False  # action tool: the LLM confirms in its own words

    parameters = {
        "name": {
            "type": "string",
            "description": "Registered name of the workflow to trigger",
            "required": True,
        },
        "message": {
            "type": "string",
            "description": "Optional free-form text passed to the workflow",
            "required": False,
            "default": "",
        },
    }

    def _registry_path(self) -> Path:
        return Path(self.config.data_dir) / REGISTRY_FILENAME

    def get_prompt_description(self) -> str:
        """List the currently registered workflow names for the text-mode prompt."""
        if not self.config:
            return super().get_prompt_description()
        registry = _load_registry(self._registry_path())
        if not registry:
            return super().get_prompt_description()

        entries = []
        for wf_name, entry in registry.items():
            desc = str(entry.get("description") or "").strip()
            entries.append(f"{wf_name} ({desc})" if desc else wf_name)
        available = ", ".join(entries)
        return (f"- {self.name}: [TOOL:{self.name}:name=NAME] - "
                f"Run a named automation. Available: {available}")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        wf_name = str(params.get("name") or "").strip()
        message = str(params.get("message") or "")

        if not self.config:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Automations are not configured.")
        if not wf_name:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Which automation should I run?")

        registry = _load_registry(self._registry_path())
        entry = registry.get(wf_name)
        if entry is None:
            failure = f"I do not have an automation called {_speakable(wf_name)}."
            known = [_speakable(n) for n in list(registry)[:3]]
            if known:
                failure += f" I know about {', '.join(known)}."
            return ToolResult(status=ToolStatus.FAILED, message=failure,
                              data={"workflow": wf_name, "delivered": False})

        session = getattr(self.assistant, "session", None)
        call_info = getattr(session, "call_info", None)
        caller = getattr(call_info, "remote_uri", "") or ""

        payload = {
            "workflow": wf_name,
            "message": message,
            "caller": caller,
            "source": "sip-agent",
        }

        try:
            # Imported here to avoid an import cycle (api.py imports the
            # tool layer at module load).
            from api import deliver_webhook
            ok = await deliver_webhook(entry["url"], payload, self.config,
                                       api_name="workflow:" + wf_name)
        except Exception as e:
            logger.error(f"Workflow trigger error for {wf_name}: {e}", exc_info=True)
            ok = False

        log_event(logger, logging.INFO,
                  f"Workflow {wf_name} trigger {'delivered' if ok else 'failed'}",
                  event="workflow_trigger", workflow=wf_name, delivered=ok)

        # data never includes the webhook URL: results can flow back through
        # the LLM and transcripts, and the URL may embed a secret token.
        if ok:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"Done. I triggered {_speakable(wf_name)}.",
                data={"workflow": wf_name, "delivered": True})
        return ToolResult(
            status=ToolStatus.FAILED,
            message="I could not reach that automation.",
            data={"workflow": wf_name, "delivered": False})
