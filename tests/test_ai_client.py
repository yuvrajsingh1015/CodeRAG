from coderag.ai_client import create_ai_client


def test_ollama_is_an_alias_for_local():
    client, provider = create_ai_client("ollama", "", "http://localhost:11434/v1/")
    assert client is not None
    assert str(client.base_url).startswith("http://localhost:11434")
    assert provider.startswith("Local backend")


def test_remote_needs_a_key():
    client, provider = create_ai_client("openai", "", "")
    assert client is None
    assert provider == "Unavailable"


def test_remote_with_key():
    client, provider = create_ai_client("remote", "sk-test", "")
    assert client is not None
    assert provider == "OpenAI"


def test_unknown_backend():
    client, _ = create_ai_client("banana", "", "")
    assert client is None
