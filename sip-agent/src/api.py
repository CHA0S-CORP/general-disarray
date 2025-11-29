"""
Outbound Call API
=================
REST API for initiating outbound notification calls with optional response collection.
"""

import asyncio
import logging
import httpx
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from main import SIPAIAssistant

logger = logging.getLogger(__name__)


def log_event(log, level, msg, event=None, **data):
    """Helper to log structured events."""
    extra = {}
    if event:
        extra['event_type'] = event
    if data:
        extra['event_data'] = data
    log.log(level, msg, extra=extra)


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


# ============================================================================
# Outbound Call Handler
# ============================================================================

class OutboundCallHandler:
    """Handles outbound notification calls."""
    
    def __init__(self, assistant: 'SIPAIAssistant'):
        self.assistant = assistant
        self.pending_calls: Dict[str, OutboundCallRequest] = {}
        self._call_counter = 0
        
    def generate_call_id(self) -> str:
        """Generate a unique call ID."""
        self._call_counter += 1
        import time
        return f"out-{int(time.time())}-{self._call_counter}"
        
    async def initiate_call(self, request: OutboundCallRequest) -> str:
        """
        Initiate an outbound call.
        Returns the call ID.
        """
        call_id = request.call_id or self.generate_call_id()
        
        log_event(logger, logging.INFO, f"Initiating outbound call to {request.extension}",
                 event="outbound_call_initiated", call_id=call_id, extension=request.extension)
        
        # Store request
        self.pending_calls[call_id] = request
        
        # Execute call in background
        asyncio.create_task(self._execute_call(call_id, request))
        
        return call_id
        
    async def _execute_call(self, call_id: str, request: OutboundCallRequest):
        """Execute the outbound call flow."""
        start_time = asyncio.get_event_loop().time()
        status = CallStatus.FAILED
        message_played = False
        choice_response = None
        choice_raw_text = None
        error = None
        
        try:
            # Build SIP URI
            extension = request.extension
            if not extension.startswith('sip:'):
                if '@' not in extension:
                    extension = f"sip:{extension}@{self.assistant.config.sip_domain}"
                else:
                    extension = f"sip:{extension}"
            
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
                raise Exception(error)
            
            status = CallStatus.RINGING
            
            # Wait for answer
            ring_start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - ring_start < request.ring_timeout:
                if getattr(call_info, 'is_active', False):
                    status = CallStatus.ANSWERED
                    break
                await asyncio.sleep(0.5)
            else:
                status = CallStatus.NO_ANSWER
                log_event(logger, logging.WARNING, f"Call not answered within {request.ring_timeout}s",
                         event="outbound_call_no_answer", call_id=call_id)
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
            
            log_event(logger, logging.INFO, "Message played",
                     event="outbound_call_message_played", call_id=call_id)
            
            # Handle choice collection if configured
            if request.choice and choice_audio:
                choice_response, choice_raw_text = await self._collect_choice(
                    call_id, call_info, request.choice, choice_audio
                )
                
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
            
            # Hang up
            if call_info.is_active:
                await self.assistant.sip_handler.hangup_call(call_info)
                
        except Exception as e:
            error = str(e)
            logger.error(f"Outbound call error: {e}", exc_info=True)
            
        finally:
            # Clean up
            if call_id in self.pending_calls:
                del self.pending_calls[call_id]
            
            # Calculate duration
            duration = asyncio.get_event_loop().time() - start_time
            
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
                
            log_event(logger, logging.INFO, f"Webhook sent successfully",
                     event="outbound_call_webhook_success", url=url)
                
        except Exception as e:
            logger.error(f"Failed to send webhook to {url}: {e}")


# ============================================================================
# FastAPI Application
# ============================================================================

def create_api(assistant: 'SIPAIAssistant') -> FastAPI:
    """Create FastAPI application for outbound calls."""
    
    app = FastAPI(
        title="SIP AI Assistant API",
        description="API for outbound notification calls with optional response collection",
        version="1.0.0"
    )
    
    handler = OutboundCallHandler(assistant)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "sip_registered": assistant.sip_handler._registered.is_set() if hasattr(assistant.sip_handler, '_registered') else False
        }
    
    @app.post("/call", response_model=OutboundCallResponse)
    async def initiate_call(request: OutboundCallRequest):
        """
        Initiate an outbound notification call.
        
        The call will be made asynchronously. If callback_url is provided,
        results will be POSTed there when the call completes.
        
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
            call_id = await handler.initiate_call(request)
            return OutboundCallResponse(
                call_id=call_id,
                status=CallStatus.QUEUED,
                message="Call initiated"
            )
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/call/{call_id}")
    async def get_call_status(call_id: str):
        """Get status of a pending call."""
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
    
    return app