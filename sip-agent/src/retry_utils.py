"""
Retry Utilities
===============
Provides configurable retry logic with exponential backoff for API calls.
"""

import asyncio
import logging
import random
from typing import TypeVar, Callable, Awaitable, Optional, Tuple, Type
from functools import wraps

from config import Config
from telemetry import Metrics

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, message: str, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.last_error = last_error


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args,
    api_name: str = "unknown",
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    config: Optional[Config] = None,
    **kwargs
) -> T:
    """
    Execute an async function with configurable retry logic.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        api_name: Name of API for logging/metrics
        max_attempts: Override config retry attempts
        base_delay: Override config base delay
        max_delay: Override config max delay
        retryable_exceptions: Tuple of exception types to retry
        config: Config instance (will use get_config() if not provided)
        **kwargs: Keyword arguments for func
        
    Returns:
        Result from successful function execution
        
    Raises:
        RetryError: If all retry attempts are exhausted
    """
    if config is None:
        from config import get_config
        config = get_config()
        
    # `is None` (not `or`): an explicit 0 is a real value. attempts is floored
    # at 1 — API_RETRY_ATTEMPTS=0 means "no retries", never "never call".
    attempts = max(1, int(config.api_retry_attempts if max_attempts is None else max_attempts))
    delay = config.api_retry_base_delay_s if base_delay is None else base_delay
    max_d = config.api_retry_max_delay_s if max_delay is None else max_delay
    
    last_error: Optional[Exception] = None
    
    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_error = e
            
            if attempt == attempts:
                # Final attempt failed
                logger.error(f"{api_name} failed after {attempts} attempts: {e}")
                raise RetryError(
                    f"{api_name} failed after {attempts} attempts",
                    last_error=e
                )
                
            # Calculate backoff with jitter
            jitter = random.uniform(0.8, 1.2)
            wait_time = min(delay * (2 ** (attempt - 1)) * jitter, max_d)
            
            logger.warning(f"{api_name} attempt {attempt} failed: {e}. Retrying in {wait_time:.2f}s")
            Metrics.record_api_retry(api_name, attempt)
            
            await asyncio.sleep(wait_time)
    
    # Should not reach here, but just in case
    raise RetryError(f"{api_name} failed", last_error=last_error)


def with_retry(
    api_name: str = "unknown",
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for adding retry logic to async functions.
    
    Usage:
        @with_retry(api_name="stt", retryable_exceptions=(httpx.HTTPError,))
        async def transcribe(audio_data: bytes) -> str:
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(
                func,
                *args,
                api_name=api_name,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_exceptions=retryable_exceptions,
                **kwargs
            )
        return wrapper
    return decorator
