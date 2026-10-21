from types import SimpleNamespace

from openai import APIConnectionError, AuthenticationError, BadRequestError, InternalServerError, RateLimitError

from coderag.retry import describe_error, is_retryable, wait_for


def _error(cls, message, status=None):
    # skip the SDK constructor, it needs a real HTTP response object
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    if status is not None:
        error.status_code = status
    return error


def _state(error, attempt=1):
    return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: error), attempt_number=attempt)


def test_wait_uses_retry_delay_from_error():
    error = _error(RateLimitError, "Quota exceeded. Please retry in 17.69s.", 429)
    assert wait_for(_state(error)) == 18.69


def test_wait_without_delay_falls_back_to_30s():
    assert wait_for(_state(_error(RateLimitError, "Too many requests", 429))) == 30.0


def test_wait_is_capped():
    assert wait_for(_state(_error(RateLimitError, "Please retry in 300s", 429))) == 65.0


def test_overloaded_waits_a_little_longer_each_time():
    error = _error(InternalServerError, "high demand", 503)
    assert wait_for(_state(error, 1)) < wait_for(_state(error, 2))


def test_what_is_retried():
    assert is_retryable(_error(RateLimitError, "slow down", 429))
    assert is_retryable(_error(InternalServerError, "high demand", 503))
    assert is_retryable(_error(APIConnectionError, "connection refused"))
    assert not is_retryable(_error(AuthenticationError, "bad key", 401))
    assert not is_retryable(_error(BadRequestError, "API key not valid", 400))
    assert not is_retryable(ValueError("bug"))


def test_readable_errors():
    assert "API key" in describe_error(_error(BadRequestError, "API key not valid. Please pass a valid API key.", 400))
    assert "34 seconds" in describe_error(_error(RateLimitError, "limit: 5. Please retry in 33.2s.", 429))
    assert "overloaded" in describe_error(_error(InternalServerError, "high demand", 503))
    assert "BASE_URL" in describe_error(_error(APIConnectionError, "connection refused"))
