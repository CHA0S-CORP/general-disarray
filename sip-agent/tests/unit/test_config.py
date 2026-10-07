"""Unit tests for Config: env-var overrides, defaults, and derived properties."""
import pytest

pytestmark = pytest.mark.unit


def test_defaults(config):
    assert config.stt_mode == "batch"
    assert config.use_realtime_stt is False
    assert config.sample_rate == 16000
    assert config.sip_user == "ai-assistant"
    assert config.llm_backend == "vllm"


def test_conversation_window_default_is_generous(config):
    """Gotcha guard: a too-small in-call history window makes the agent 'forget'
    what was said earlier in the same call (a live .env had it set to 1). The
    shipped default must keep a usable amount of context."""
    assert config.max_conversation_turns >= 10


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


def test_thinking_switch_parsing(config, config_factory):
    """Unset = None (send nothing); true/false parse case-insensitively."""
    assert config.llm_enable_thinking is None
    assert config_factory(llm_enable_thinking="false").llm_enable_thinking is False
    assert config_factory(llm_enable_thinking="TRUE").llm_enable_thinking is True
    assert config_factory(llm_enable_thinking="0").llm_enable_thinking is False
    assert config_factory(llm_enable_thinking="").llm_enable_thinking is None


def test_split_system_prompt_default_off(config, config_factory):
    """Off by default: Qwen3.5's template rejects a later system message and
    gpt-oss drops it. Opt-in only."""
    assert config.llm_split_system_prompt is False
    assert config_factory(llm_split_system_prompt="true").llm_split_system_prompt is True
    assert config_factory(llm_split_system_prompt="1").llm_split_system_prompt is True


# --- boolean env parsing (_env_bool) ----------------------------------------

@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On", " on "])
def test_env_bool_truthy_values(config_factory, raw):
    assert config_factory(knowledge_enabled=raw).knowledge_enabled is True
    assert config_factory(knowledge_auto_inject=raw).knowledge_auto_inject is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "Off"])
def test_env_bool_falsy_values(config_factory, raw):
    assert config_factory(knowledge_enabled=raw).knowledge_enabled is False
    assert config_factory(knowledge_auto_inject=raw).knowledge_auto_inject is False


@pytest.mark.parametrize("raw", ["", "   ", "maybe", "enabled"])
def test_env_bool_blank_or_garbage_keeps_default(config_factory, raw):
    # default-on stays on, default-off stays off
    assert config_factory(knowledge_enabled=raw).knowledge_enabled is True
    assert config_factory(knowledge_auto_inject=raw).knowledge_auto_inject is False


def test_env_bool_helper_direct(monkeypatch):
    from config import _env_bool
    monkeypatch.delenv("GD_TEST_FLAG", raising=False)
    assert _env_bool("GD_TEST_FLAG", True) is True
    assert _env_bool("GD_TEST_FLAG", False) is False
    monkeypatch.setenv("GD_TEST_FLAG", "yes")
    assert _env_bool("GD_TEST_FLAG", False) is True
    monkeypatch.setenv("GD_TEST_FLAG", "0")
    assert _env_bool("GD_TEST_FLAG", True) is False


def test_no_bool_field_uses_lower_eq_true_parsing():
    """Regression guard: `.lower() == "true"` silently treats 1/yes/on as False."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "src" / "config.py").read_text()
    assert not re.search(r'os\.getenv\([^)]*\)\.lower\(\)\s*==\s*"true"', src)


def test_tool_enable_flags(config, config_factory):
    assert config.enable_timer_tool is True
    assert config.enable_callback_tool is True
    assert config.enable_weather_tool is True
    assert config_factory(enable_timer_tool="no").enable_timer_tool is False
    assert config_factory(enable_weather_tool="0").enable_weather_tool is False


def test_new_policy_defaults(config):
    assert config.voice_dial_allow_pattern == ""
    assert config.voice_dial_deny_pattern
    assert config.callback_max_per_call == 3
    assert config.callback_max_delay_s == 86400
    assert config.stt_timeout_s == 15.0
    assert config.speech_reprobe_interval_s == 15.0
    assert config.verify_lockout_failures == 5
    assert config.verify_lockout_s == 900


def test_unused_tool_flags_removed(config):
    assert not hasattr(config, "enable_search_tool")
    assert not hasattr(config, "enable_calendar_tool")
