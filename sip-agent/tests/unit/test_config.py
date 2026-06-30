"""Unit tests for Config: env-var overrides, defaults, and derived properties."""
import pytest

pytestmark = pytest.mark.unit


def test_defaults(config):
    assert config.stt_mode == "batch"
    assert config.use_realtime_stt is False
    assert config.sample_rate == 16000
    assert config.sip_user == "ai-assistant"
    assert config.llm_backend == "vllm"


def test_whisper_api_url_alias(config):
    assert config.whisper_api_url == config.speaches_api_url


def test_realtime_mode_toggle(config_factory):
    cfg = config_factory(stt_mode="realtime")
    assert cfg.stt_mode == "realtime"
    assert cfg.use_realtime_stt is True


def test_service_url_overrides(config_factory):
    cfg = config_factory(
        speaches_api_url="http://speaches.test:9001",
        llm_base_url="http://vllm.test:9000/v1",
    )
    assert cfg.speaches_api_url == "http://speaches.test:9001"
    assert cfg.whisper_api_url == "http://speaches.test:9001"
    assert cfg.llm_base_url == "http://vllm.test:9000/v1"


def test_outbound_sip_uri_flag_parsing(config_factory):
    assert config_factory().outbound_allow_sip_uri is False
    assert config_factory(outbound_allow_sip_uri="true").outbound_allow_sip_uri is True


def test_phrases_cache_is_deduplicated(config):
    phrases = config.phrases.get_all_phrases_for_cache()
    assert len(phrases) == len(set(phrases))
    assert any("Hello" in p or "Hi" in p for p in phrases)
