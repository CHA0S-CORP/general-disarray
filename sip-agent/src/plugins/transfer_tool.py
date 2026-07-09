"""
Transfer Tool Plugin
====================
Blind-transfers the live caller to another extension via SIP REFER.

The target is validated against the same outbound dial policy as the REST
API (`api.check_extension_allowed`): raw ``sip:`` URIs are only accepted
when OUTBOUND_ALLOW_SIP_URI is set, and OUTBOUND_EXTENSION_PATTERN can
restrict extensions further. Bare extensions are expanded to
``sip:<extension>@<SIP_DOMAIN>``.

Usage in conversation:
User: "Can you put me through to the front desk?"
LLM: "I'm transferring you to the front desk now. [TOOL:TRANSFER:extension=2001]"
"""

import logging
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)


class TransferTool(BaseTool):
    """Blind-transfer the current caller to another extension (SIP REFER)."""

    name = "TRANSFER"
    description = (
        "Transfer the current caller to another extension. Tell the caller "
        "who you are transferring them to before using this."
    )
    enabled = True
    speak_result = False  # action tool: the LLM announces the transfer itself

    parameters = {
        "extension": {
            "type": "string",
            "description": "Extension number to transfer the caller to "
                           "(or a full sip: URI when policy allows)",
            "required": True,
        },
    }

    def __init__(self, assistant):
        super().__init__(assistant)
        if self.config and not self.config.enable_transfer_tool:
            self.enabled = False
            logger.info("TRANSFER tool disabled - ENABLE_TRANSFER_TOOL is false")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        # Imported inside execute to avoid an import cycle with api.py.
        from api import check_extension_allowed

        extension = str(params.get("extension") or "").strip()

        if not self.assistant or not getattr(self.assistant, "current_call", None):
            return ToolResult(
                status=ToolStatus.FAILED,
                message="There is no active call to transfer.",
            )

        # Same dial-target policy as the REST /call endpoint, so the voice
        # path cannot be talked into routing calls to arbitrary SIP domains.
        policy_error = check_extension_allowed(extension, self.config)
        if policy_error:
            log_event(logger, logging.WARNING,
                      f"TRANSFER target rejected by policy: {policy_error}",
                      event="transfer_target_blocked",
                      extension=extension, reason=policy_error)
            return ToolResult(
                status=ToolStatus.FAILED,
                message="I cannot transfer to that number.",
            )

        if extension.startswith("sip:"):
            target = extension
        else:
            target = f"sip:{extension}@{self.config.sip_domain}"

        handler = getattr(self.assistant, "sip_handler", None)
        if handler is None or not hasattr(handler, "transfer_call"):
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Transfers are not available.",
            )

        try:
            ok = await handler.transfer_call(self.assistant.current_call, target)
        except Exception as e:  # transfer_call shouldn't raise, but never crash a call
            logger.error(f"Transfer error: {e}", exc_info=True)
            ok = False

        if ok:
            log_event(logger, logging.INFO, f"Transferring caller to {target}",
                      event="transfer_initiated", extension=extension, target=target)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message="Transferring you now.",
                data={"extension": extension, "target": target, "transferred": True},
            )

        return ToolResult(
            status=ToolStatus.FAILED,
            message="I could not complete the transfer.",
        )
