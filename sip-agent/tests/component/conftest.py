"""Fixtures for the component tier: in-process mock Speaches/vLLM servers and a
minimal fake assistant that wires the real tool manager.

The mock servers run as real ASGI apps under uvicorn on ephemeral localhost
ports, so the agent's real httpx / AsyncOpenAI clients hit a real URL (closer to
production than monkeypatching the transport).
"""
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn

# Make the mock app modules importable.
sys.path.insert(0, str(Path(__file__).parent / "mocks"))
import mock_speaches  # noqa: E402
import mock_vllm  # noqa: E402

pytestmark = pytest.mark.component


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ThreadedServer:
    """Run a uvicorn server in a background thread for the test session."""

    def __init__(self, app):
        self.port = _free_port()
        cfg = uvicorn.Config(app, host="127.0.0.1", port=self.port,
                             log_level="warning", lifespan="off")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()
        for _ in range(200):  # up to ~10s
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("mock server failed to start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture(scope="session")
def speaches_url():
    srv = ThreadedServer(mock_speaches.build_app())
    srv.start()
    yield srv.url
    srv.stop()


@pytest.fixture(scope="session")
def vllm_url():
    srv = ThreadedServer(mock_vllm.build_app())
    srv.start()
    yield srv.url
    srv.stop()


@pytest.fixture
def comp_config(config_factory, speaches_url, vllm_url):
    """Config pointed at the mock servers (config_factory comes from root conftest)."""
    return config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
    )


class StubLLMEngine:
    """Recording stub for the reformat_for_speech API surface."""

    def __init__(self):
        self.reformat_calls = []

    async def reformat_for_speech(self, text, timeout_s):
        self.reformat_calls.append(text)
        return f"SPOKEN {text}"


class FakeAssistant:
    """Minimal stand-in for SIPAIAssistant: real ToolManager, stub SIP."""

    def __init__(self, config):
        self.config = config
        self.current_call = None
        # No `_registered` attr -> /health reports sip_registered: False.
        self.sip_handler = SimpleNamespace()
        self.audio_pipeline = None
        self.llm_engine = StubLLMEngine()
        self.scheduled_callbacks = []
        # Import here so `src` is on sys.path (set by the root conftest).
        from tool_manager import ToolManager
        from transcript_store import TranscriptStore
        self.tool_manager = ToolManager(self)
        self.transcripts = TranscriptStore(config)

    async def schedule_callback(self, delay, message, destination):
        self.scheduled_callbacks.append((delay, message, destination))
        return "cb-test"


@pytest.fixture
def assistant(comp_config):
    return FakeAssistant(comp_config)


@pytest.fixture
def client(assistant):
    """FastAPI TestClient over the real create_api() app (no real SIP/phone)."""
    from fastapi.testclient import TestClient
    from api import create_api
    app = create_api(assistant, call_queue=None)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def make_client():
    """Factory: build a (TestClient, FakeAssistant) for a given Config.

    Used by tests that need a non-default config (e.g. API_AUTH_TOKEN set).
    """
    from fastapi.testclient import TestClient
    from api import create_api

    created = []

    def _make(config):
        a = FakeAssistant(config)
        c = TestClient(create_api(a, call_queue=None))
        c.__enter__()
        created.append(c)
        return c, a

    yield _make
    for c in created:
        c.__exit__(None, None, None)
