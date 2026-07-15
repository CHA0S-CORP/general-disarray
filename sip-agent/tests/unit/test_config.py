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


def test_call_event_defaults(config):
    assert config.call_event_webhook_url == ""
    assert config.call_events == "call.started,call.ended"
    assert config.call_event_include_transcript is True


def test_call_event_overrides(config_factory):
    cfg = config_factory(
        call_event_webhook_url="http://n8n:5678/webhook/x/webhook",
        call_events="call.ended",
        call_event_include_transcript="false",
    )
    assert cfg.call_event_webhook_url == "http://n8n:5678/webhook/x/webhook"
    assert cfg.call_events == "call.ended"
    assert cfg.call_event_include_transcript is False


def test_turn_ack_mode(config, config_factory):
    assert config.turn_ack_mode == "chime"
    assert config_factory(turn_ack_mode="phrase").turn_ack_mode == "phrase"
    assert config_factory(turn_ack_mode="NONE").turn_ack_mode == "none"
    assert config_factory(turn_ack_mode="kazoo").turn_ack_mode == "chime"  # invalid -> fallback


def test_endpoint_mode(config, config_factory):
    assert config.endpoint_mode == "fixed"
    assert config_factory(endpoint_mode="adaptive").endpoint_mode == "adaptive"
    assert config_factory(endpoint_mode="SPECULATIVE").endpoint_mode == "speculative"
    assert config_factory(endpoint_mode="psychic").endpoint_mode == "fixed"  # invalid -> fallback


def test_endpoint_silence_bounds(config, config_factory):
    assert config.endpoint_min_silence_ms == 350
    assert config.endpoint_max_silence_ms == 1500
    cfg = config_factory(endpoint_min_silence_ms="250", endpoint_max_silence_ms="2000")
    assert cfg.endpoint_min_silence_ms == 250
    assert cfg.endpoint_max_silence_ms == 2000


def test_chime_volume(config, config_factory):
    assert config.chime_volume == 0.3
    assert config_factory(chime_volume="0.15").chime_volume == 0.15
    assert config_factory(chime_volume="7").chime_volume == 1.0  # clamped


def test_llm_frequency_penalty(config, config_factory):
    assert config.llm_frequency_penalty == 0.0
    assert config_factory(llm_frequency_penalty="0.5").llm_frequency_penalty == 0.5


def test_system_prompt_default(config):
    assert "General Disarray" in config.system_prompt
    assert "[TOOL:" not in config.system_prompt


def test_system_prompt_env_override(config_factory):
    assert config_factory(system_prompt="Custom voice bot.").system_prompt == "Custom voice bot."


def test_system_prompt_empty_env_falls_back(config_factory):
    assert "General Disarray" in config_factory(system_prompt="").system_prompt


def test_system_prompt_file_wins_over_env(config_factory, tmp_path):
    (tmp_path / "system_prompt.txt").write_text("From the file.")
    cfg = config_factory(data_dir=str(tmp_path), system_prompt="From the env.")
    assert cfg.system_prompt == "From the file."


def test_acknowledgments_removed(config):
    assert not hasattr(config.phrases, "acknowledgments")
    assert "Okay." not in config.phrases.get_all_phrases_for_cache()


def test_stale_acknowledgments_key_in_phrases_json_is_ignored(config_factory, tmp_path):
    import json as _json
    (tmp_path / "phrases.json").write_text(_json.dumps({
        "greetings": ["Yo."],
        "acknowledgments": ["Okay."],
    }))
    cfg = config_factory(data_dir=str(tmp_path))
    assert cfg.phrases.greetings == ["Yo."]
    assert "Okay." not in cfg.phrases.get_all_phrases_for_cache()


def test_message_reformat_timeout(config, config_factory):
    assert config.message_reformat_timeout_s == 20.0
    assert config_factory(message_reformat_timeout_s="3.5").message_reformat_timeout_s == 3.5


def test_tool_round_budget(config, config_factory):
    assert config.llm_max_tool_rounds == 5
    assert config.llm_agent_timeout_s == 30.0
    assert config_factory(llm_max_tool_rounds="2").llm_max_tool_rounds == 2
    assert config_factory(llm_agent_timeout_s="12.5").llm_agent_timeout_s == 12.5


def test_caller_memory_defaults_and_overrides(config, config_factory):
    assert config.caller_memory_enabled is True
    assert config.caller_memory_max_facts == 15
    assert config.caller_memory_max_chars == 1500
    assert config.caller_memory_timeout_s == 30.0
    assert config_factory(caller_memory_enabled="false").caller_memory_enabled is False


def test_knowledge_defaults_and_dir_resolution(config, config_factory, tmp_path):
    assert config.knowledge_enabled is True
    assert config.knowledge_auto_inject is False
    assert config.knowledge_chunk_size == 800
    assert config.knowledge_chunk_overlap == 120
    assert config.knowledge_top_k == 3
    # Default dir resolves under data_dir; explicit env wins.
    assert config.knowledge_dir == config.data_dir / "knowledge"
    cfg = config_factory(knowledge_dir=str(tmp_path / "kb"))
    assert str(cfg.knowledge_dir) == str(tmp_path / "kb")


def test_summary_defaults(config, config_factory):
    assert config.summary_enabled is True
    assert config.summary_timeout_s == 20.0
    assert config_factory(
        conversation_summary_enabled="false").summary_enabled is False


def test_caller_memory_dir_created(config):
    assert (config.data_dir / "caller_memory").is_dir()
