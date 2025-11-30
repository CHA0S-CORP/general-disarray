"""
Outbound Call API
=================
REST API for initiating outbound notification calls with optional response collection.
"""

import asyncio
import logging
import httpx
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, model_validator

from telemetry import create_span, Metrics
from logging_utils import log_event

if TYPE_CHECKING:
    from main import SIPAIAssistant
    from call_queue import CallQueue

logger = logging.getLogger(__name__)


# ============================================================================
# API Models
# ============================================================================

class ChoiceOption(BaseModel):
    """A choice option for the user."""
    value: str = Field(..., description="The value to return if selected")
    synonyms: List[str] = Field(default_factory=list, description="Alternative phrases that map to this choice")


class ChoicePrompt(BaseModel):
    """Configuration for collecting user choice."""
    prompt: str = Field(..., description="Question to ask the user")
    options: List[ChoiceOption] = Field(..., description="Valid choice options")
    timeout_seconds: int = Field(default=30, description="How long to wait for response")
    repeat_count: int = Field(default=2, description="How many times to repeat prompt if no response")


class OutboundCallRequest(BaseModel):
    """Request to initiate an outbound notification call."""
    message: str = Field(..., description="Message to speak to the recipient")
    extension: str = Field(..., description="SIP extension or phone number to call")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL to POST results to (required if choice is specified)")
    ring_timeout: int = Field(default=30, description="Seconds to wait for call to be answered")
    choice: Optional[ChoicePrompt] = Field(default=None, description="Optional choice prompt for collecting response")
    call_id: Optional[str] = Field(default=None, description="Optional caller-provided ID for tracking")
    
    @model_validator(mode='after')
    def validate_callback_url_required_for_choice(self):
        """Validate that callback_url is provided when choice is specified."""
        if self.choice is not None and not self.callback_url:
            raise ValueError("callback_url is required when choice is specified")
        return self


class CallStatus(str, Enum):
    """Status of an outbound call."""
    QUEUED = "queued"
    RINGING = "ringing"
    ANSWERED = "answered"
    COMPLETED = "completed"
    NO_ANSWER = "no_answer"
    FAILED = "failed"
    BUSY = "busy"


class OutboundCallResponse(BaseModel):
    """Response to outbound call request."""
    call_id: str
    status: CallStatus
    message: str
    queue_position: Optional[int] = None


class WebhookPayload(BaseModel):
    """Payload sent to callback webhook."""
    call_id: str
    status: CallStatus
    extension: str
    duration_seconds: float
    message_played: bool
    choice_response: Optional[str] = None
    choice_raw_text: Optional[str] = None
    error: Optional[str] = None


class ToolExecuteRequest(BaseModel):
    """Request to execute a tool."""
    tool: str = Field(..., description="Name of the tool to execute (e.g., WEATHER, DATETIME)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the tool")
    speak_result: bool = Field(default=False, description="Speak the result to the active call")
    call_id: Optional[str] = Field(default=None, description="Specific call to speak to (if multiple calls active)")


class ToolCallRequest(BaseModel):
    """Request to execute a tool and call someone with the result."""
    tool: str = Field(..., description="Name of the tool to execute (e.g., WEATHER, DATETIME)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the tool")
    extension: str = Field(..., description="SIP extension or phone number to call")
    prefix: Optional[str] = Field(default=None, description="Message to speak before the tool result")
    suffix: Optional[str] = Field(default=None, description="Message to speak after the tool result")
    ring_timeout: int = Field(default=30, description="Seconds to wait for call to be answered")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL to POST call results to")


class ToolCallResponse(BaseModel):
    """Response from tool call request."""
    call_id: str
    status: str
    tool: str
    tool_success: bool
    tool_message: str
    message: str


class ScheduledCallRequest(BaseModel):
    """Request to schedule a call for a future time."""
    extension: str = Field(..., description="SIP extension or phone number to call")
    message: Optional[str] = Field(default=None, description="Message to speak (if no tool specified)")
    tool: Optional[str] = Field(default=None, description="Tool to execute and speak result (e.g., WEATHER)")
    tool_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    delay_seconds: Optional[int] = Field(default=None, description="Seconds from now to make the call")
    at_time: Optional[str] = Field(default=None, description="ISO datetime or HH:MM time to make the call")
    timezone: Optional[str] = Field(default="America/Los_Angeles", description="Timezone for at_time (default: America/Los_Angeles)")
    prefix: Optional[str] = Field(default=None, description="Message to speak before tool result")
    suffix: Optional[str] = Field(default=None, description="Message to speak after tool result")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL to POST results to")
    recurring: Optional[str] = Field(default=None, description="Recurrence pattern: 'daily', 'weekdays', 'weekends', or cron expression")
    
    @model_validator(mode='after')
    def validate_time_or_delay(self):
        """Validate that either delay_seconds or at_time is provided."""
        if self.delay_seconds is None and self.at_time is None:
            raise ValueError("Either delay_seconds or at_time must be provided")
        if self.delay_seconds is not None and self.at_time is not None:
            raise ValueError("Provide either delay_seconds or at_time, not both")
        return self
    
    @model_validator(mode='after')
    def validate_message_or_tool(self):
        """Validate that either message or tool is provided."""
        if not self.message and not self.tool:
            raise ValueError("Either message or tool must be provided")
        return self


class ScheduledCallResponse(BaseModel):
    """Response from scheduled call request."""
    schedule_id: str
    status: str
    extension: str
    scheduled_for: str
    delay_seconds: int
    message: str
    recurring: Optional[str] = None


class ScheduledCallInfo(BaseModel):
    """Information about a scheduled call."""
    schedule_id: str
    extension: str
    scheduled_for: str
    remaining_seconds: int
    message: Optional[str] = None
    tool: Optional[str] = None
    recurring: Optional[str] = None
    status: str


class ToolExecuteResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    tool: str
    message: str
    data: Optional[Dict[str, Any]] = None
    spoken: bool = False
    error: Optional[str] = None


class ToolInfo(BaseModel):
    """Information about an available tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool


# ============================================================================
# Outbound Call Handler
# ============================================================================

class OutboundCallHandler:
    """Handles outbound notification calls."""
    
    def __init__(self, assistant: 'SIPAIAssistant', call_queue: 'CallQueue' = None):
        self.assistant = assistant
        self.call_queue = call_queue
        self.pending_calls: Dict[str, OutboundCallRequest] = {}
        self._call_counter = 0
        
    def generate_call_id(self) -> str:
        """Generate a unique call ID."""
        self._call_counter += 1
        import time
        return f"out-{int(time.time())}-{self._call_counter}"
        
    async def initiate_call(self, request: OutboundCallRequest) -> tuple[str, int]:
        """
        Initiate an outbound call.
        Returns (call_id, queue_position).
        """
        call_id = request.call_id or self.generate_call_id()
        
        log_event(logger, logging.INFO, f"Initiating outbound call to {request.extension}",
                 event="outbound_call_initiated", call_id=call_id, extension=request.extension)
        
        # Use queue if available
        if self.call_queue:
            queued_call = await self.call_queue.enqueue(call_id, request)
            return call_id, queued_call.position
        else:
            # Direct execution (no queue)
            self.pending_calls[call_id] = request
            asyncio.create_task(self._execute_call(call_id, request))
            return call_id, 0
        
    async def _execute_call(self, call_id: str, request: OutboundCallRequest):
        """Execute the outbound call flow."""
        start_time = asyncio.get_event_loop().time()
        status = CallStatus.FAILED
        message_played = False
        choice_response = None
        choice_raw_text = None
        error = None
        
        with create_span("api.execute_call", {
            "call.id": call_id,
            "call.extension": request.extension,
            "call.has_choice": request.choice is not None
        }) as span:
            try:
                # Build SIP URI
                extension = request.extension
                if not extension.startswith('sip:'):
                    if '@' not in extension:
                        extension = f"sip:{extension}@{self.assistant.config.sip_domain}"
                    else:
                        extension = f"sip:{extension}"
                
                span.set_attribute("call.sip_uri", extension)
                log_event(logger, logging.INFO, f"Making call to {extension}",
                         event="outbound_call_dialing", call_id=call_id, uri=extension)
                
                # Pre-generate TTS for message
                message_audio = await self.assistant.audio_pipeline.synthesize(request.message)
                if not message_audio:
                    raise Exception("Failed to generate TTS for message")
                
                # Pre-generate TTS for choice prompt if needed
                choice_audio = None
                if request.choice:
                    choice_audio = await self.assistant.audio_pipeline.synthesize(request.choice.prompt)
                
                # Make the call
                call_info = await self.assistant.sip_handler.make_call(extension)
                if not call_info:
                    status = CallStatus.FAILED
                    error = "Failed to initiate call"
                    span.set_attribute("call.error", error)
                    Metrics.record_call_failed("outbound", "initiate_failed")
                    raise Exception(error)
                
                status = CallStatus.RINGING
                span.set_attribute("call.status", "ringing")
                
                # Wait for answer
                ring_start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - ring_start < request.ring_timeout:
                    if getattr(call_info, 'is_active', False):
                        status = CallStatus.ANSWERED
                        span.set_attribute("call.status", "answered")
                        break
                    await asyncio.sleep(0.5)
                else:
                    status = CallStatus.NO_ANSWER
                    span.set_attribute("call.status", "no_answer")
                    log_event(logger, logging.WARNING, f"Call not answered within {request.ring_timeout}s",
                             event="outbound_call_no_answer", call_id=call_id)
                    Metrics.record_call_failed("outbound", "no_answer")
                    await self.assistant.sip_handler.hangup_call(call_info)
                    raise Exception("Call not answered")
                
                log_event(logger, logging.INFO, "Call answered",
                         event="outbound_call_answered", call_id=call_id)
                
                # Wait for media to be ready
                await asyncio.sleep(1)
                
                # Play the message
                await self.assistant.sip_handler.send_audio(call_info, message_audio)
                audio_duration = len(message_audio) / (self.assistant.config.sample_rate * 2)
                await asyncio.sleep(audio_duration + 0.5)
                message_played = True
                span.set_attribute("call.message_played", True)
                
                log_event(logger, logging.INFO, "Message played",
                         event="outbound_call_message_played", call_id=call_id)
                
                # Handle choice collection if configured
                if request.choice and choice_audio:
                    choice_response, choice_raw_text = await self._collect_choice(
                        call_id, call_info, request.choice, choice_audio
                    )
                    
                    span.set_attribute("call.choice_response", choice_response or "none")
                    log_event(logger, logging.INFO, f"Choice collected: {choice_response}",
                             event="outbound_call_choice_collected", call_id=call_id, 
                             response=choice_response, raw_text=choice_raw_text)
                    
                    # Play acknowledgment if choice was matched
                    if choice_response and call_info.is_active:
                        try:
                            ack_audio = await self.assistant.audio_pipeline.synthesize("Acknowledged.")
                            if ack_audio:
                                await self.assistant.sip_handler.send_audio(call_info, ack_audio)
                                ack_duration = len(ack_audio) / (self.assistant.config.sample_rate * 2)
                                await asyncio.sleep(ack_duration + 0.3)
                                log_event(logger, logging.INFO, "Acknowledgment played",
                                         event="outbound_call_ack_played", call_id=call_id)
                        except Exception as e:
                            logger.warning(f"Failed to play acknowledgment: {e}")
                
                status = CallStatus.COMPLETED
                span.set_attribute("call.status", "completed")
                
                # Hang up
                if call_info.is_active:
                    await self.assistant.sip_handler.hangup_call(call_info)
                    
            except Exception as e:
                error = str(e)
                logger.error(f"Outbound call error: {e}", exc_info=True)
                span.record_exception(e)
                span.set_attribute("call.error", error)
                
            finally:
                # Clean up
                if call_id in self.pending_calls:
                    del self.pending_calls[call_id]
                
                # Calculate duration
                duration = asyncio.get_event_loop().time() - start_time
                span.set_attribute("call.duration_s", round(duration, 2))
                
                # Send webhook if callback_url provided
                if request.callback_url:
                    await self._send_webhook(
                        request.callback_url,
                        WebhookPayload(
                            call_id=call_id,
                            status=status,
                            extension=request.extension,
                            duration_seconds=round(duration, 2),
                            message_played=message_played,
                            choice_response=choice_response,
                            choice_raw_text=choice_raw_text,
                            error=error
                        )
                    )
            
    async def _collect_choice(
        self, 
        call_id: str,
        call_info,
        choice: ChoicePrompt,
        choice_audio: bytes
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Collect user choice via voice.
        Returns (matched_value, raw_transcription).
        """
        for attempt in range(choice.repeat_count):
            # Play prompt
            await self.assistant.sip_handler.send_audio(call_info, choice_audio)
            audio_duration = len(choice_audio) / (self.assistant.config.sample_rate * 2)
            await asyncio.sleep(audio_duration + 0.3)
            
            # Listen for response
            response_text = await self._listen_for_response(
                call_info, 
                timeout=choice.timeout_seconds
            )
            
            if response_text:
                # Try to match to a choice
                matched = self._match_choice(response_text, choice.options)
                if matched:
                    return matched, response_text
                    
                # No match - will retry if attempts remain
                log_event(logger, logging.INFO, f"No choice matched for: {response_text}",
                         event="outbound_call_choice_no_match", call_id=call_id, 
                         attempt=attempt + 1, text=response_text)
        
        return None, response_text if response_text else None
        
    async def _listen_for_response(self, call_info, timeout: float) -> Optional[str]:
        """Listen for user speech and return transcription."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if not getattr(call_info, 'is_active', False):
                break
                
            if not getattr(call_info, 'media_ready', False):
                await asyncio.sleep(0.1)
                continue
                
            try:
                audio_chunk = await self.assistant.sip_handler.receive_audio(
                    call_info, 
                    timeout=0.1
                )
                
                if audio_chunk:
                    transcription = await self.assistant.audio_pipeline.process_audio(audio_chunk)
                    if transcription and len(transcription.strip()) > 1:
                        return transcription.strip()
                        
            except Exception as e:
                logger.debug(f"Audio receive error: {e}")
                
            await asyncio.sleep(0.05)
            
        return None
        
    def _match_choice(self, text: str, options: List[ChoiceOption]) -> Optional[str]:
        """Match transcribed text to a choice option."""
        text_lower = text.lower().strip()
        
        for option in options:
            # Check exact value match
            if option.value.lower() == text_lower:
                return option.value
                
            # Check synonyms
            for synonym in option.synonyms:
                if synonym.lower() in text_lower or text_lower in synonym.lower():
                    return option.value
                    
            # Check if value is contained in text
            if option.value.lower() in text_lower:
                return option.value
                
        return None
        
    async def _send_webhook(self, url: str, payload: WebhookPayload):
        """Send result to callback webhook."""
        with create_span("api.send_webhook", {
            "webhook.url": url,
            "webhook.call_id": payload.call_id,
            "webhook.status": payload.status.value
        }) as span:
            try:
                log_event(logger, logging.INFO, f"Sending webhook to {url}",
                         event="outbound_call_webhook", url=url, status=payload.status)
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        url,
                        json=payload.model_dump(),
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    
                span.set_attribute("webhook.success", True)
                span.set_attribute("http.status_code", response.status_code)
                log_event(logger, logging.INFO, f"Webhook sent successfully",
                         event="outbound_call_webhook_success", url=url)
                
                # Record success metric
                Metrics.record_callback_success()
                    
            except httpx.TimeoutException as e:
                logger.error(f"Webhook timeout to {url}: {e}")
                span.set_attribute("webhook.success", False)
                span.set_attribute("error.type", "timeout")
                Metrics.record_callback_failed("timeout")
            except httpx.HTTPStatusError as e:
                logger.error(f"Webhook HTTP error to {url}: {e}")
                span.set_attribute("webhook.success", False)
                span.set_attribute("http.status_code", e.response.status_code)
                span.record_exception(e)
                Metrics.record_callback_failed(f"http_{e.response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send webhook to {url}: {e}")
                span.set_attribute("webhook.success", False)
                span.record_exception(e)
                # Record failure metric
                Metrics.record_callback_failed(type(e).__name__)


# ============================================================================
# FastAPI Application
# ============================================================================

def create_api(assistant: 'SIPAIAssistant', call_queue: 'CallQueue' = None) -> FastAPI:
    """Create FastAPI application for outbound calls."""
    
    app = FastAPI(
        title="SIP AI Assistant API",
        description="API for outbound notification calls with optional response collection",
        version="1.0.0"
    )
    
    handler = OutboundCallHandler(assistant, call_queue)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        result = {
            "status": "healthy",
            "sip_registered": assistant.sip_handler._registered.is_set() if hasattr(assistant.sip_handler, '_registered') else False
        }
        if call_queue:
            result["queue"] = await call_queue.get_queue_status()
        return result
    
    @app.get("/queue")
    async def queue_status():
        """Get call queue status."""
        if not call_queue:
            return {"enabled": False}
        
        status = await call_queue.get_queue_status()
        return {
            "enabled": True,
            **status
        }
    
    @app.post("/call", response_model=OutboundCallResponse)
    async def initiate_call(request: OutboundCallRequest):
        """
        Initiate an outbound notification call.
        
        The call will be made asynchronously. If callback_url is provided,
        results will be POSTed there when the call completes.
        
        Calls are queued and processed sequentially to prevent overwhelming the SIP system.
        
        Note: callback_url is required when using choice collection.
        
        Simple notification example:
        ```json
        {
            "message": "Hello, this is a reminder about your appointment.",
            "extension": "1001"
        }
        ```
        
        Choice collection example (requires callback_url):
        ```json
        {
            "message": "Hello, this is a reminder about your appointment tomorrow at 2pm.",
            "extension": "1001",
            "callback_url": "https://example.com/webhook",
            "choice": {
                "prompt": "Say yes to confirm or no to cancel.",
                "options": [
                    {"value": "confirmed", "synonyms": ["yes", "yeah", "yep", "confirm"]},
                    {"value": "cancelled", "synonyms": ["no", "nope", "cancel"]}
                ],
                "timeout_seconds": 15
            }
        }
        ```
        """
        try:
            call_id, position = await handler.initiate_call(request)
            return OutboundCallResponse(
                call_id=call_id,
                status=CallStatus.QUEUED,
                message=f"Call queued at position {position}" if position > 0 else "Call initiated",
                queue_position=position if position > 0 else None
            )
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/call/{call_id}")
    async def get_call_status(call_id: str):
        """Get status of a call."""
        # Check queue first
        if call_queue:
            queued_call = await call_queue.get_call(call_id)
            if queued_call:
                return {
                    "call_id": call_id,
                    "status": queued_call.status.value,
                    "queued_at": queued_call.queued_at,
                    "started_at": queued_call.started_at,
                    "completed_at": queued_call.completed_at,
                    "error": queued_call.error
                }
        
        # Check pending calls (direct execution mode)
        if call_id in handler.pending_calls:
            return {
                "call_id": call_id,
                "status": "in_progress",
                "extension": handler.pending_calls[call_id].extension
            }
            
        return {
            "call_id": call_id,
            "status": "not_found"
        }
    
    # Store handler reference for queue worker
    app.state.handler = handler
    
    # ==========================================================================
    # Tool Execution API
    # ==========================================================================
    
    @app.get("/tools", response_model=List[ToolInfo])
    async def list_tools():
        """
        List all available tools.
        
        Returns information about each tool including name, description, 
        parameters, and whether it's enabled.
        """
        tools = []
        for name, tool in assistant.tool_manager.tools.items():
            tools.append(ToolInfo(
                name=name,
                description=getattr(tool, 'description', ''),
                parameters=getattr(tool, 'parameters', {}),
                enabled=getattr(tool, 'enabled', True)
            ))
        return sorted(tools, key=lambda t: t.name)
    
    @app.get("/tools/{tool_name}", response_model=ToolInfo)
    async def get_tool(tool_name: str):
        """Get information about a specific tool."""
        tool = assistant.tool_manager.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        return ToolInfo(
            name=getattr(tool, 'name', tool_name),
            description=getattr(tool, 'description', ''),
            parameters=getattr(tool, 'parameters', {}),
            enabled=getattr(tool, 'enabled', True)
        )
    
    @app.post("/tools/{tool_name}/call", response_model=ToolCallResponse)
    async def tool_call(tool_name: str, request: ToolCallRequest):
        """
        Execute a tool and call someone with the result.
        
        This endpoint is perfect for webhooks - it executes a tool (like WEATHER),
        then places an outbound call to speak the result to the recipient.
        
        Examples:
        
        Weather alert call:
        ```json
        POST /tools/WEATHER/call
        {
            "tool": "WEATHER",
            "extension": "1001",
            "prefix": "Good morning! Here's your weather update.",
            "suffix": "Have a great day!"
        }
        ```
        
        Scheduled weather call (from cron/Home Assistant):
        ```bash
        curl -X POST http://sip-agent:8080/tools/WEATHER/call \\
          -H "Content-Type: application/json" \\
          -d '{"tool": "WEATHER", "extension": "5551234567"}'
        ```
        
        DateTime announcement:
        ```json
        POST /tools/DATETIME/call
        {
            "tool": "DATETIME",
            "params": {"format": "full"},
            "extension": "1001",
            "prefix": "Attention please."
        }
        ```
        
        With callback for confirmation:
        ```json
        {
            "tool": "WEATHER",
            "extension": "1001",
            "callback_url": "https://example.com/webhook/weather-call-complete"
        }
        ```
        """
        actual_tool_name = tool_name.upper()
        
        # Get the tool
        tool = assistant.tool_manager.get_tool(actual_tool_name)
        if not tool:
            raise HTTPException(
                status_code=404, 
                detail=f"Tool '{actual_tool_name}' not found. Use GET /tools to list available tools."
            )
        
        log_event(logger, logging.INFO, f"Tool call request: {actual_tool_name} -> {request.extension}",
                 event="api_tool_call", tool=actual_tool_name, extension=request.extension)
        
        try:
            # Execute the tool first
            result = await tool.execute(request.params)
            
            tool_success = result.status.value == "success" if hasattr(result.status, 'value') else str(result.status).lower() == "success"
            tool_message = result.message
            
            if not tool_success:
                log_event(logger, logging.WARNING, f"Tool failed: {tool_message}",
                         event="api_tool_call_tool_failed", tool=actual_tool_name)
                return ToolCallResponse(
                    call_id="",
                    status="tool_failed",
                    tool=actual_tool_name,
                    tool_success=False,
                    tool_message=tool_message,
                    message=f"Tool execution failed: {tool_message}"
                )
            
            # Build the full message
            message_parts = []
            if request.prefix:
                message_parts.append(request.prefix)
            message_parts.append(tool_message)
            if request.suffix:
                message_parts.append(request.suffix)
            
            full_message = " ".join(message_parts)
            
            # Create outbound call request
            call_request = OutboundCallRequest(
                message=full_message,
                extension=request.extension,
                callback_url=request.callback_url,
                ring_timeout=request.ring_timeout
            )
            
            # Initiate the call
            call_id, position = await handler.initiate_call(call_request)
            
            log_event(logger, logging.INFO, f"Tool call initiated: {call_id}",
                     event="api_tool_call_initiated", 
                     tool=actual_tool_name, 
                     call_id=call_id,
                     extension=request.extension)
            
            return ToolCallResponse(
                call_id=call_id,
                status="queued" if position > 0 else "initiated",
                tool=actual_tool_name,
                tool_success=True,
                tool_message=tool_message,
                message=f"Calling {request.extension} with {actual_tool_name} result"
            )
            
        except Exception as e:
            logger.error(f"Tool call failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/tools/{tool_name}/execute", response_model=ToolExecuteResponse)
    async def execute_tool(tool_name: str, request: ToolExecuteRequest = None):
        """
        Execute a tool and optionally speak the result.
        
        This endpoint allows external systems (webhooks, home automation, etc.)
        to trigger tool execution. The result can optionally be spoken to 
        an active call.
        
        Examples:
        
        Get weather (just data):
        ```json
        POST /tools/WEATHER/execute
        {"tool": "WEATHER"}
        ```
        
        Get weather and speak to call:
        ```json
        POST /tools/WEATHER/execute
        {"tool": "WEATHER", "speak_result": true}
        ```
        
        Execute calculation:
        ```json
        POST /tools/CALC/execute
        {"tool": "CALC", "params": {"expression": "25 * 4"}}
        ```
        
        Set a timer and announce it:
        ```json
        POST /tools/SET_TIMER/execute
        {
            "tool": "SET_TIMER",
            "params": {"duration": 300, "message": "Pizza is ready!"},
            "speak_result": true
        }
        ```
        """
        # Use request body or default
        if request is None:
            request = ToolExecuteRequest(tool=tool_name)
        
        # Override tool name from path
        actual_tool_name = tool_name.upper()
        
        # Get the tool
        tool = assistant.tool_manager.get_tool(actual_tool_name)
        if not tool:
            raise HTTPException(
                status_code=404, 
                detail=f"Tool '{actual_tool_name}' not found. Use GET /tools to list available tools."
            )
        
        log_event(logger, logging.INFO, f"API executing tool: {actual_tool_name}",
                 event="api_tool_execute", tool=actual_tool_name, params=request.params)
        
        try:
            # Execute the tool
            result = await tool.execute(request.params)
            
            success = result.status.value == "success" if hasattr(result.status, 'value') else result.status == "success"
            
            response = ToolExecuteResponse(
                success=success,
                tool=actual_tool_name,
                message=result.message,
                data=getattr(result, 'data', None),
                spoken=False,
                error=None if success else result.message
            )
            
            # Speak result to active call if requested
            if request.speak_result and success and result.message:
                spoken = await _speak_to_call(assistant, result.message, request.call_id)
                response.spoken = spoken
                if not spoken:
                    log_event(logger, logging.WARNING, "No active call to speak to",
                             event="api_tool_no_call")
            
            log_event(logger, logging.INFO, f"Tool executed: {actual_tool_name}",
                     event="api_tool_complete", tool=actual_tool_name, success=success)
            
            return response
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/speak")
    async def speak_message(message: str, call_id: Optional[str] = None):
        """
        Speak a message to the active call.
        
        This is useful for external systems to inject announcements
        into an ongoing call.
        
        Query params:
        - message: The text to speak
        - call_id: Optional specific call ID (if multiple calls active)
        """
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        spoken = await _speak_to_call(assistant, message, call_id)
        
        if spoken:
            return {"success": True, "message": "Message spoken to call"}
        else:
            raise HTTPException(status_code=404, detail="No active call to speak to")
    
    # ==========================================================================
    # Scheduled Calls API
    # ==========================================================================
    
    @app.post("/schedule", response_model=ScheduledCallResponse)
    async def schedule_call(request: ScheduledCallRequest):
        """
        Schedule a call for a future time.
        
        You can schedule a call with either a static message or a tool that
        will be executed at call time (e.g., WEATHER for fresh data).
        
        Time can be specified as:
        - delay_seconds: Number of seconds from now
        - at_time: ISO datetime (2025-01-15T07:00:00) or HH:MM time (07:00)
        
        Examples:
        
        Wake-up weather call in 8 hours:
        ```json
        {
            "extension": "1001",
            "tool": "WEATHER",
            "delay_seconds": 28800,
            "prefix": "Good morning! Here's your weather."
        }
        ```
        
        Daily 7am weather call:
        ```json
        {
            "extension": "1001",
            "tool": "WEATHER",
            "at_time": "07:00",
            "timezone": "America/Los_Angeles",
            "prefix": "Good morning!",
            "recurring": "daily"
        }
        ```
        
        Reminder call at specific time:
        ```json
        {
            "extension": "5551234567",
            "message": "This is your reminder to take your medication.",
            "at_time": "2025-01-15T09:00:00",
            "timezone": "America/New_York"
        }
        ```
        
        Weekday morning briefing:
        ```json
        {
            "extension": "1001",
            "tool": "WEATHER",
            "at_time": "06:30",
            "recurring": "weekdays",
            "prefix": "Good morning! Time to wake up."
        }
        ```
        """
        import time
        import pytz
        from datetime import datetime, timedelta
        
        # Calculate delay
        delay_seconds = request.delay_seconds
        scheduled_time = None
        
        if request.at_time:
            try:
                tz = pytz.timezone(request.timezone or "America/Los_Angeles")
                now = datetime.now(tz)
                
                # Parse time - either full ISO or just HH:MM
                if 'T' in request.at_time or '-' in request.at_time:
                    # Full ISO datetime
                    if request.at_time.endswith('Z'):
                        scheduled_time = datetime.fromisoformat(request.at_time.replace('Z', '+00:00'))
                    else:
                        scheduled_time = datetime.fromisoformat(request.at_time)
                        if scheduled_time.tzinfo is None:
                            scheduled_time = tz.localize(scheduled_time)
                else:
                    # Just HH:MM - schedule for today or tomorrow
                    hour, minute = map(int, request.at_time.split(':'))
                    scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If time already passed today, schedule for tomorrow
                    if scheduled_time <= now:
                        scheduled_time += timedelta(days=1)
                
                delay_seconds = int((scheduled_time - now).total_seconds())
                
                if delay_seconds < 0:
                    raise HTTPException(status_code=400, detail="Scheduled time is in the past")
                    
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid time format: {e}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing time: {e}")
        
        # Generate schedule ID
        schedule_id = f"sched-{int(time.time())}-{len(assistant.tool_manager.scheduled_tasks) + 1}"
        
        # Build the task data
        task_data = {
            "extension": request.extension,
            "message": request.message,
            "tool": request.tool,
            "tool_params": request.tool_params,
            "prefix": request.prefix,
            "suffix": request.suffix,
            "callback_url": request.callback_url,
            "recurring": request.recurring,
            "timezone": request.timezone,
            "at_time": request.at_time,  # Store for recurring
        }
        
        # Schedule the task
        task_id = await assistant.tool_manager.schedule_task(
            task_type="scheduled_call",
            delay_seconds=delay_seconds,
            message=request.message or f"Scheduled {request.tool} call",
            target_uri=request.extension,
            metadata=task_data
        )
        
        # Calculate scheduled time for response
        if scheduled_time:
            scheduled_for = scheduled_time.isoformat()
        else:
            tz = pytz.timezone(request.timezone or "America/Los_Angeles")
            scheduled_for = (datetime.now(tz) + timedelta(seconds=delay_seconds)).isoformat()
        
        log_event(logger, logging.INFO, f"Scheduled call: {schedule_id} -> {request.extension}",
                 event="call_scheduled",
                 schedule_id=schedule_id,
                 extension=request.extension,
                 delay=delay_seconds,
                 tool=request.tool)
        
        return ScheduledCallResponse(
            schedule_id=task_id,
            status="scheduled",
            extension=request.extension,
            scheduled_for=scheduled_for,
            delay_seconds=delay_seconds,
            message=f"Call scheduled for {scheduled_for}",
            recurring=request.recurring
        )
    
    @app.get("/schedule", response_model=List[ScheduledCallInfo])
    async def list_scheduled_calls():
        """List all scheduled calls."""
        scheduled = []
        now = asyncio.get_event_loop().time()
        
        for task_id, task in assistant.tool_manager.scheduled_tasks.items():
            if task.task_type == "scheduled_call":
                remaining = max(0, int(task.execute_at - now))
                metadata = task.metadata or {}
                
                scheduled.append(ScheduledCallInfo(
                    schedule_id=task_id,
                    extension=metadata.get("extension", task.target_uri or ""),
                    scheduled_for=datetime.fromtimestamp(task.execute_at).isoformat(),
                    remaining_seconds=remaining,
                    message=metadata.get("message"),
                    tool=metadata.get("tool"),
                    recurring=metadata.get("recurring"),
                    status="pending" if not task.completed else "completed"
                ))
        
        return sorted(scheduled, key=lambda x: x.remaining_seconds)
    
    @app.get("/schedule/{schedule_id}", response_model=ScheduledCallInfo)
    async def get_scheduled_call(schedule_id: str):
        """Get details of a scheduled call."""
        task = assistant.tool_manager.scheduled_tasks.get(schedule_id)
        
        if not task or task.task_type != "scheduled_call":
            raise HTTPException(status_code=404, detail="Scheduled call not found")
        
        now = asyncio.get_event_loop().time()
        remaining = max(0, int(task.execute_at - now))
        metadata = task.metadata or {}
        
        return ScheduledCallInfo(
            schedule_id=schedule_id,
            extension=metadata.get("extension", task.target_uri or ""),
            scheduled_for=datetime.fromtimestamp(task.execute_at).isoformat(),
            remaining_seconds=remaining,
            message=metadata.get("message"),
            tool=metadata.get("tool"),
            recurring=metadata.get("recurring"),
            status="pending" if not task.completed else "completed"
        )
    
    @app.delete("/schedule/{schedule_id}")
    async def cancel_scheduled_call(schedule_id: str):
        """Cancel a scheduled call."""
        task = assistant.tool_manager.scheduled_tasks.get(schedule_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Scheduled call not found")
        
        # Remove from scheduled tasks
        del assistant.tool_manager.scheduled_tasks[schedule_id]
        
        log_event(logger, logging.INFO, f"Cancelled scheduled call: {schedule_id}",
                 event="call_schedule_cancelled", schedule_id=schedule_id)
        
        return {"success": True, "message": f"Scheduled call {schedule_id} cancelled"}
    
    return app


async def _speak_to_call(assistant: 'SIPAIAssistant', message: str, call_id: Optional[str] = None) -> bool:
    """
    Speak a message to an active call.
    
    Returns True if message was spoken, False if no active call.
    """
    try:
        # Check if there's an active call
        current_call = getattr(assistant, 'current_call', None)
        
        if not current_call:
            logger.debug("No current_call attribute on assistant")
            return False
        
        # If call_id specified, verify it matches
        if call_id:
            current_call_id = getattr(current_call, 'call_id', None) or getattr(current_call, 'id', None)
            if current_call_id != call_id:
                logger.debug(f"Call ID mismatch: {current_call_id} != {call_id}")
                return False
        
        # Generate TTS
        audio_data = await assistant.audio_pipeline.synthesize(message)
        if not audio_data:
            logger.error("Failed to synthesize speech")
            return False
        
        # Send audio to call
        await assistant.sip_handler.send_audio(audio_data)
        
        log_event(logger, logging.INFO, f"Spoke message to call: {message[:50]}...",
                 event="api_speak_success")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to speak to call: {e}", exc_info=True)
        return False