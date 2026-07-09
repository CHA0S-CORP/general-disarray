"""
Call Queue
==========
Redis-backed queue for outbound calls with concurrency control.
Prevents overwhelming the SIP system when multiple calls are requested.
"""

import os
import time
import json
import asyncio
import logging
from typing import Optional, Dict, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

import redis.asyncio as redis

if TYPE_CHECKING:
    from api import OutboundCallRequest, OutboundCallHandler

from telemetry import Metrics
from logging_utils import log_event

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    """Timezone-aware UTC timestamp (datetime.utcnow is deprecated)."""
    return datetime.now(timezone.utc).isoformat()


class QueuedCallStatus(str, Enum):
    """Status of a queued call."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedCall:
    """A call in the queue."""
    call_id: str
    request_json: str  # Serialized OutboundCallRequest
    status: QueuedCallStatus
    queued_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    position: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueuedCall':
        data['status'] = QueuedCallStatus(data['status'])
        return cls(**data)


class CallQueue:
    """
    Redis-backed call queue with concurrency control.
    
    Uses Redis for:
    - Queue persistence (survives restarts)
    - Atomic operations (safe for concurrent access)
    - Call status tracking
    
    Configuration via environment variables:
    - REDIS_URL: Full Redis URL (default: redis://localhost:6379/0)
    - REDIS_PASSWORD: Redis password for authentication (optional)
    - REDIS_SSL: Set to "true" to enable SSL/TLS (optional)
    """
    
    QUEUE_KEY = "sip:call_queue"
    PROCESSING_KEY = "sip:call_processing"
    CALL_PREFIX = "sip:call:"
    
    def __init__(
        self,
        redis_url: str = None,
        max_concurrent: int = 1
    ):
        # Build Redis URL with authentication if provided
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis_password = os.environ.get("REDIS_PASSWORD")
        self.redis_ssl = os.environ.get("REDIS_SSL", "").lower() == "true"
        
        self.max_concurrent = max_concurrent
        self.redis: Optional[redis.Redis] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._handler: Optional['OutboundCallHandler'] = None
        self._running = False
        self._semaphore: Optional[asyncio.Semaphore] = None
        # Strong references to in-flight call-processing tasks so they are not
        # garbage-collected mid-call and can be cancelled/awaited on stop().
        self._tasks: Set[asyncio.Task] = set()
        
    async def connect(self):
        """Connect to Redis with optional authentication."""
        # Build connection kwargs
        connect_kwargs = {
            "decode_responses": True,
        }
        
        if self.redis_password:
            connect_kwargs["password"] = self.redis_password
            logger.info("Redis authentication enabled")
            
        if self.redis_ssl:
            connect_kwargs["ssl"] = True
            logger.info("Redis SSL/TLS enabled")
        
        self.redis = redis.from_url(self.redis_url, **connect_kwargs)
        await self.redis.ping()
        
        # Log connection without exposing password
        safe_url = self.redis_url.split("@")[-1] if "@" in self.redis_url else self.redis_url
        logger.info(f"Connected to Redis at {safe_url}")
        
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            
    async def start(self, handler: 'OutboundCallHandler'):
        """Start the queue worker."""
        self._handler = handler
        self._running = True
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Requeue any calls that were processing when we stopped
        await self._recover_processing_calls()
        
        # Start worker
        self._worker_task = asyncio.create_task(self._worker_loop())
        
        log_event(logger, logging.INFO, f"Call queue started (max_concurrent={self.max_concurrent})",
                 event="queue_started", max_concurrent=self.max_concurrent)
        
    async def stop(self):
        """Stop the queue worker."""
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Cancel and await any in-flight call-processing tasks BEFORE the
        # caller disconnects Redis, so each task's finally-block can clear the
        # PROCESSING_KEY entry and persist final status while Redis is still up.
        if self._tasks:
            pending = list(self._tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        logger.info("Call queue stopped")
        
    async def enqueue(self, call_id: str, request: 'OutboundCallRequest') -> QueuedCall:
        """Add a call to the queue."""
        queued_call = QueuedCall(
            call_id=call_id,
            request_json=request.model_dump_json(),
            status=QueuedCallStatus.QUEUED,
            queued_at=_utcnow_iso()
        )
        
        # Store call data
        await self.redis.set(
            f"{self.CALL_PREFIX}{call_id}",
            json.dumps(queued_call.to_dict()),
            ex=86400  # Expire after 24 hours
        )
        
        # Add to queue - rpush returns new list length (our position)
        position = await self.redis.rpush(self.QUEUE_KEY, call_id)
        queued_call.position = position
        
        # Record queue metrics
        Metrics.record_queue_enqueued()
        Metrics.record_queue_depth(position)
        
        log_event(logger, logging.INFO, f"Call {call_id} queued at position {position}",
                 event="call_queued", call_id=call_id, position=position)
        
        return queued_call
        
    async def get_call(self, call_id: str) -> Optional[QueuedCall]:
        """Get call status."""
        data = await self.redis.get(f"{self.CALL_PREFIX}{call_id}")
        if data:
            return QueuedCall.from_dict(json.loads(data))
        return None
        
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_length = await self.redis.llen(self.QUEUE_KEY)
        processing_count = await self.redis.scard(self.PROCESSING_KEY)
        
        return {
            "queued": queue_length,
            "processing": processing_count,
            "max_concurrent": self.max_concurrent
        }
        
    async def _recover_processing_calls(self):
        """Requeue calls that were processing during shutdown."""
        processing = await self.redis.smembers(self.PROCESSING_KEY)
        
        for call_id in processing:
            logger.warning(f"Recovering interrupted call: {call_id}")
            # Move back to queue
            await self.redis.lpush(self.QUEUE_KEY, call_id)
            await self.redis.srem(self.PROCESSING_KEY, call_id)
            
            # Update status
            call = await self.get_call(call_id)
            if call:
                call.status = QueuedCallStatus.QUEUED
                call.started_at = None
                await self.redis.set(
                    f"{self.CALL_PREFIX}{call_id}",
                    json.dumps(call.to_dict()),
                    ex=86400
                )
                
    async def _worker_loop(self):
        """Process calls from the queue."""
        while self._running:
            try:
                # Acquire a processing slot BEFORE pulling a call from the queue
                # so a call is only popped (and marked processing) when a slot is
                # actually free. This keeps the PROCESSING_KEY count accurate and
                # avoids draining the whole queue into the processing set at once.
                slot_start = time.monotonic()
                await self._semaphore.acquire()
                wait_time = time.monotonic() - slot_start

                try:
                    # Try to get a call from queue (blocking with timeout)
                    result = await self.redis.blpop(self.QUEUE_KEY, timeout=1)

                    if result is None:
                        # No call available; release the slot and retry.
                        self._semaphore.release()
                        continue

                    _, call_id = result

                    # Mark as processing
                    await self.redis.sadd(self.PROCESSING_KEY, call_id)

                    # Spawn the processing task (it releases the slot when done).
                    # Keep a strong ref so it isn't GC'd and can be awaited on stop.
                    task = asyncio.create_task(
                        self._process_call_with_semaphore(call_id, wait_time)
                    )
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                except BaseException:
                    # If we failed before handing the slot to a task, release it
                    # so the slot is not leaked (re-raise for the handlers below).
                    self._semaphore.release()
                    raise

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _process_call_with_semaphore(self, call_id: str, wait_time: float = 0.0):
        """Process a call and release the processing slot when finished.

        The semaphore is acquired by the worker loop before the call is popped;
        this task owns it and releases it in the finally block.
        """
        try:
            wait_time_ms = wait_time * 1000

            # Record queue wait time metric
            Metrics.record_queue_wait_time(wait_time_ms)

            if wait_time > 0.1:  # Log if waited more than 100ms
                log_event(logger, logging.INFO, f"Call {call_id} waited {wait_time:.1f}s for slot",
                         event="call_waited", call_id=call_id, wait_seconds=round(wait_time, 1))

            await self._process_call(call_id)
        finally:
            self._semaphore.release()
                
    async def _process_call(self, call_id: str):
        """Process a single call."""
        call = None
        try:
            # Get call data
            call = await self.get_call(call_id)
            if not call:
                logger.error(f"Call {call_id} not found in queue")
                return
                
            # Update status
            call.status = QueuedCallStatus.PROCESSING
            call.started_at = _utcnow_iso()
            await self.redis.set(
                f"{self.CALL_PREFIX}{call_id}",
                json.dumps(call.to_dict()),
                ex=86400
            )
            
            log_event(logger, logging.INFO, f"Processing call {call_id}",
                     event="call_processing", call_id=call_id)
            
            # Deserialize request
            from api import OutboundCallRequest
            request = OutboundCallRequest.model_validate_json(call.request_json)
            
            # Execute the call. Expected failures (initiate failure, no
            # answer) don't raise — the final CallStatus is returned instead,
            # so persist the real outcome rather than blanket COMPLETED.
            # A handler returning None (e.g. a test stub) counts as success,
            # matching the old no-exception-means-completed contract.
            result = await self._handler._execute_call(call_id, request)
            final_status, error = result if result is not None else (None, None)

            if final_status is None or final_status.value == "completed":
                call.status = QueuedCallStatus.COMPLETED
            else:
                call.status = QueuedCallStatus.FAILED
                call.error = error or f"Call ended with status '{final_status.value}'"
            call.completed_at = _utcnow_iso()
            
        except Exception as e:
            logger.error(f"Call {call_id} failed: {e}", exc_info=True)
            if call:
                call.status = QueuedCallStatus.FAILED
                call.error = str(e)
                call.completed_at = _utcnow_iso()
                
        finally:
            # Remove from processing set
            await self.redis.srem(self.PROCESSING_KEY, call_id)
            
            # Update final status
            if call:
                await self.redis.set(
                    f"{self.CALL_PREFIX}{call_id}",
                    json.dumps(call.to_dict()),
                    ex=86400
                )