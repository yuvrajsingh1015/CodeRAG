"""Retry rules shared by embedding and chat calls."""

import logging
import re

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_backoff = wait_exponential(multiplier=0.5, max=8)  # 0.5, 1, 2, 4, 8 seconds


def _retry_delay(error: Exception) -> float | None:
    """Free tiers answer 429 with a hint like "Please retry in 17.69s"."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) + 1 if match else None


def _is_overloaded(error: BaseException) -> bool:
    return isinstance(error, APIStatusError) and error.status_code in (500, 502, 503, 504)


def is_retryable(error: BaseException) -> bool:
    """Rate limits, overloaded servers and network problems can pass, a wrong key or model will not."""
    return isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)) or _is_overloaded(error)


def wait_for(state: RetryCallState) -> float:
    error = state.outcome.exception() if state.outcome else None
    if isinstance(error, RateLimitError):
        delay = _retry_delay(error) or 30.0
        logger.warning("Rate limited, waiting %.0fs", delay)
        return min(delay, 65.0)
    if _is_overloaded(error):
        logger.warning("Model overloaded, trying again")
        return 2.0 * state.attempt_number
    return _backoff(state)


def api_retry(attempts: int):
    return retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(attempts),
        wait=wait_for,
        reraise=True,  # re-raise the last exception after all retries are exhausted
    )


def describe_error(error: Exception) -> str:
    """One readable sentence instead of the raw JSON error body."""
    # Gemini answers a wrong key with 400 "API key not valid"
    bad_key = isinstance(error, APIStatusError) and "api key" in str(error).lower()
    if isinstance(error, (AuthenticationError, PermissionDeniedError)) or bad_key:
        return "The API key was rejected. Check API_KEY in your .env."
    if isinstance(error, NotFoundError):
        return "The model was not found. Check CHAT_MODEL and EMBEDDING_MODEL in your .env."
    if isinstance(error, RateLimitError):
        delay = _retry_delay(error)
        wait = f" Try again in about {delay:.0f} seconds." if delay else " Try again in a minute."
        return "The API rate limit was reached (free tiers allow only a few requests per minute)." + wait
    if _is_overloaded(error):
        return "The model is overloaded right now. Try again in a moment or pick another CHAT_MODEL."
    if isinstance(error, APIConnectionError):
        return "Could not reach the AI backend. Is it running, and is BASE_URL right?"
    return str(error)[:300]
