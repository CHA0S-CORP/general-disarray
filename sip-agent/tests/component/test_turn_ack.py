"""Component tests for the per-turn acknowledgment (TURN_ACK_MODE gating)."""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.component


def _call(uri="sip:1001@host"):
    return SimpleNamespace(is_active=True, remote_uri=uri, media_ready=False)


def _assistant(config_factory, speaches_url, vllm_url, **overrides):
    from main import SIPAIAssistant
    cfg = config_factory(
        speaches_api_url=speaches_url,
        llm_base_url=f"{vllm_url}/v1",
        llm_model="mock-model",
        **overrides,
    )
    return SIPAIAssistant(cfg)


@pytest.fixture
def sent_audio(monkeypatch):
    """Recorder factory: patch an assistant's sip_handler.send_audio."""
    payloads = []

    def attach(assistant):
        async def fake_send(call, audio):
            payloads.append(audio)
            return True
        monkeypatch.setattr(assistant.sip_handler, "send_audio", fake_send)
        return payloads

    return attach


async def test_chime_mode_plays_chime_first(config_factory, speaches_url, vllm_url, sent_audio):
    a = _assistant(config_factory, speaches_url, vllm_url)  # default mode = chime
    payloads = sent_audio(a)
    session = a._begin_session(_call(), "inbound", "sip:1001@host")

    await a._run_turn(session, "hello there")

    assert payloads, "expected the chime to be sent"
    assert payloads[0] == a._chime_pcm
    await a._teardown_session()


async def test_none_mode_sends_no_chime(config_factory, speaches_url, vllm_url, sent_audio):
    a = _assistant(config_factory, speaches_url, vllm_url, turn_ack_mode="none")
    payloads = sent_audio(a)
    session = a._begin_session(_call(), "inbound", "sip:1001@host")

    await a._run_turn(session, "hello there")

    assert a._chime_pcm not in payloads
    await a._teardown_session()


async def test_phrase_mode_speaks_thinking_phrase(config_factory, speaches_url, vllm_url, monkeypatch):
    a = _assistant(config_factory, speaches_url, vllm_url, turn_ack_mode="phrase")
    spoken = []

    async def fake_speak(text):
        spoken.append(text)

    monkeypatch.setattr(a, "_speak", fake_speak)
    session = a._begin_session(_call(), "inbound", "sip:1001@host")

    await a._run_turn(session, "hello there")

    assert spoken, "expected a spoken filler"
    assert spoken[0] in a.thinking_phrases
    await a._teardown_session()
