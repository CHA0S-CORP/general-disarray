"""
Retry Utilities
===============
Provides configurable retry logic with exponential backoff for API calls.
"""

import asyncio
import logging
import random
from typing import TypeVar, Callable, Awaitable, Optional, Tuple, Type, TYPE_CHECKING
from functools import wraps

from config import Config
from telemetry import Metrics

if TYPE_CHECKING:
    import httpx

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
        
    attempts = max_attempts or config.api_retry_attempts
    delay = base_delay or config.api_retry_base_delay_s
    max_d = max_delay or config.api_retry_max_delay_s
    
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


class RetryableHTTPClient:
    """
    HTTP client wrapper with built-in retry logic.
    
    This provides a shared httpx client with retry capabilities
    to avoid creating new clients for each request.
    """
    
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            from config import get_config
            config = get_config()
        self.config = config
        self._client: Optional['httpx.AsyncClient'] = None
        
    async def get_client(self) -> 'httpx.AsyncClient':
        """Get or create the httpx client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.api_timeout_s),
                follow_redirects=True,
            )
        return self._client
        
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            
    async def post(
        self,
        url: str,
        api_name: str = "http",
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        **kwargs
    ):
        """
        Make a POST request with retry logic.
        
        Args:
            url: Request URL
            api_name: Name for logging/metrics
            retryable_exceptions: Exception types to retry
            **kwargs: Arguments passed to httpx.post
        """
        import httpx
        
        if retryable_exceptions is None:
            retryable_exceptions = (
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
            )
            
        async def do_post():
            client = await self.get_client()
            response = await client.post(url, **kwargs)
            response.raise_for_status()
            return response
            
        return await retry_async(
            do_post,
            api_name=api_name,
            retryable_exceptions=retryable_exceptions,
            config=self.config,
        )
        
    async def get(
        self,
        url: str,
        api_name: str = "http",
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        **kwargs
    ):
        """Make a GET request with retry logic."""
        import httpx
        
        if retryable_exceptions is None:
            retryable_exceptions = (
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
            )
            
        async def do_get():
            client = await self.get_client()
            response = await client.get(url, **kwargs)
            response.raise_for_status()
            return response
            
        return await retry_async(
            do_get,
            api_name=api_name,
            retryable_exceptions=retryable_exceptions,
            config=self.config,
        )


# Global shared HTTP client instance
_shared_http_client: Optional[RetryableHTTPClient] = None


def get_http_client(config: Optional[Config] = None) -> RetryableHTTPClient:
    """Get the shared HTTP client instance."""
    global _shared_http_client
    if _shared_http_client is None:
        _shared_http_client = RetryableHTTPClient(config)
    return _shared_http_client


async def close_http_client():
    """Close the shared HTTP client."""
    global _shared_http_client
    if _shared_http_client:
        await _shared_http_client.close()
        _shared_http_client = None
