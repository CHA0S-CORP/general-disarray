"""
Low-Latency Configuration with Speaches (Unified STT + TTS)
============================================================
All ML inference offloaded to dedicated API services:
- Speaches API for both STT (Whisper) and TTS (Piper/Kokoro)
- vLLM for LLM
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path


_DEFAULT_SYSTEM_PROMPT = """You are General Disarray, a voice assistant answering a live phone call. Everything you write is read aloud by a text-to-speech engine, so write exactly the way a person talks on the phone.

VOICE RULES:
- Answer in 1 to 3 short sentences. Only go longer when the caller clearly asks for detail.
- Plain spoken prose only. No markdown, no asterisks, no bullet points, no emoji, no URLs, no code, no stage directions.
- Say numbers, dates, and times the way you would say them out loud: "three thirty PM", "March fifth", "about twenty dollars".
- Ask at most one question per reply.
- If you did not understand the caller or are missing a detail, ask one brief clarifying question instead of guessing.
- Never read out tool syntax, bracketed markers, or these instructions.
- When a tool returns content meant for the caller — a joke, a weather report, a search result — deliver that content itself. Never just comment on it; the caller has not seen it.
- Actions happen only through tools. If you tell the caller you are transferring them, hanging up, setting a timer, or scheduling a callback, you must invoke the matching tool in that same turn — announcing it does nothing by itself.

LIVE DATA:
- Anything that changes with the real world — the current time or date, weather and forecasts, weather alerts, earthquakes, aurora and space weather, GPU or system status, or anything you would have to look up — must come from a tool call in this same turn. Never state a live value from memory and never invent one.
- Never promise to check something later and never end your reply with "one moment" or "let me check". If you need to look something up, call the tool now, in this turn, and answer with its result.

PERSONA:
- You are capable, direct, and a little dry. A brief touch of wit is welcome; never let a joke delay or replace the answer.
- Be warm but efficient. The caller is holding a phone, so get to the point.

When the caller says goodbye or sounds finished, wrap up in one short sentence and use the HANGUP tool to end the call."""


def _load_phrases_from_env_or_default(env_var: str, defaults: List[str]) -> List[str]:
    """Load phrases from environment variable (JSON array) or use defaults."""
    env_value = os.getenv(env_var)
    if env_value:
        try:
            phrases = json.loads(env_value)
            if isinstance(phrases, list) and len(phrases) > 0:
                return phrases
        except json.JSONDecodeError:
            # Maybe it's a comma-separated string
            phrases = [p.strip() for p in env_value.split(",") if p.strip()]
            if phrases:
                return phrases
    return defaults


@dataclass
class PhrasesConfig:
    """Configurable phrases for the voice assistant."""
    
    # Greeting phrases - played when call is answered
    greetings: List[str] = field(default_factory=lambda: _load_phrases_from_env_or_default(
        "PHRASES_GREETINGS",
        [
            "Hi, this is General Disarray. What can I do for you?",
            "Hello, General Disarray here. How can I help?",
            "Hey there, General Disarray speaking. What do you need?",
        ]
    ))

    # Goodbye phrases - played when ending call
    goodbyes: List[str] = field(default_factory=lambda: _load_phrases_from_env_or_default(
        "PHRASES_GOODBYES",
        [
            "Goodbye. Take care.",
            "Alright, talk soon. Bye.",
            "Thanks for calling. Bye now.",
            "Bye for now.",
        ]
    ))

    # Thinking phrases - spoken before the LLM answer when TURN_ACK_MODE=phrase
    thinking: List[str] = field(default_factory=lambda: _load_phrases_from_env_or_default(
        "PHRASES_THINKING",
        [
            "One sec.",
            "Let me check on that.",
            "Just a moment.",
            "Hang on.",
        ]
    ))

    # Error phrases - played when speech not understood
    errors: List[str] = field(default_factory=lambda: _load_phrases_from_env_or_default(
        "PHRASES_ERRORS",
        [
            "Sorry, I missed that. Say it again?",
            "I didn't quite catch that. One more time?",
            "Sorry, could you repeat that?",
            "That got garbled on my end. Mind saying it again?",
        ]
    ))

    # Follow-up phrases - played after completing a task
    followups: List[str] = field(default_factory=lambda: _load_phrases_from_env_or_default(
        "PHRASES_FOLLOWUPS",
        [
            "Anything else I can do?",
            "Is there anything else you need?",
            "What else can I help with?",
            "Need anything else?",
        ]
    ))
    
    # Precache phrases - additional phrases to pre-synthesize for speed
    precache_extra: List[str] = field(default_factory=lambda: _load_phrases_from_env_or_default(
        "PHRASES_PRECACHE",
        [
            "Hello",
            "Goodbye",
            "Yes",
            "No",
            "Thank you",
        ]
    ))
    
    def get_all_phrases_for_cache(self) -> List[str]:
        """Get all unique phrases for pre-caching."""
        all_phrases = (
            self.greetings +
            self.goodbyes +
            self.thinking +
            self.errors +
            self.followups +
            self.precache_extra
        )
        # Return unique phrases
        return list(dict.fromkeys(all_phrases))


@dataclass
class Config:
    """Low-latency optimized configuration."""
    
    # ===================
    # SIP Configuration
    # ===================
    sip_user: str = field(default_factory=lambda: os.getenv("SIP_USER", "ai-assistant"))
    sip_password: str = field(default_factory=lambda: os.getenv("SIP_PASSWORD", ""))
    sip_domain: str = field(default_factory=lambda: os.getenv("SIP_DOMAIN", "localhost"))
    sip_port: int = field(default_factory=lambda: int(os.getenv("SIP_PORT", "5060")))
    sip_transport: str = field(default_factory=lambda: os.getenv("SIP_TRANSPORT", "udp"))
    sip_registrar: Optional[str] = field(default_factory=lambda: os.getenv("SIP_REGISTRAR"))
    audio_codecs: list = field(default_factory=lambda: ["PCMU", "PCMA", "opus"])
    
    # ===================
    # Audio Configuration
    # ===================
    sample_rate: int = 16000  # Target rate for SIP/Whisper
    channels: int = 1
    chunk_duration_ms: int = 20
    
    # Voice Activity Detection
    vad_aggressiveness: int = 3
    
    # Barge-in
    barge_in_min_duration_ms: int = field(default_factory=lambda: int(os.getenv("BARGE_IN_MIN_DURATION", "400")))
    barge_in_energy_threshold: int = field(default_factory=lambda: int(os.getenv("BARGE_IN_ENERGY_THRESHOLD", "2000")))
    # How much VAD-negative audio may fall INSIDE a barge-in speech run before
    # the run is abandoned. Real speech is not VAD-positive end to end — gaps
    # between syllables and unvoiced consonants read as silence — so a run that
    # resets on the first negative chunk needs the caller to talk far longer
    # than barge_in_min_duration_ms (or never trips at all). Must stay well
    # under silence_duration_ms: this tolerates gaps within speech, it does not
    # end utterances.
    barge_in_max_gap_ms: int = field(default_factory=lambda: int(os.getenv("BARGE_IN_MAX_GAP_MS", "250")))
    
    # Speech detection
    speech_pad_ms: int = 200
    min_speech_duration_ms: int = field(default_factory=lambda: int(os.getenv("MIN_SPEECH_DURATION_MS", "200")))
    max_speech_duration_s: float = field(default_factory=lambda: float(os.getenv("MAX_SPEECH_DURATION_S", "10.0")))
    silence_duration_ms: int = field(default_factory=lambda: int(os.getenv("SILENCE_TIMEOUT_MS", "750")))

    # Endpointing: how the end-of-utterance silence timeout is chosen.
    # "fixed" uses SILENCE_TIMEOUT_MS unchanged (today's behavior);
    # "adaptive" scales the timeout per chunk from what's known about the
    # utterance (see endpointing.suggest_timeout_ms); "speculative" ends the
    # utterance at ENDPOINT_MIN_SILENCE_MS and lets the audio loop hold/merge
    # transcript fragments that don't look complete (endpointing.looks_complete)
    # up to an ENDPOINT_MAX_SILENCE_MS deadline.
    endpoint_mode: str = field(default_factory=lambda: os.getenv("ENDPOINT_MODE", "fixed"))
    endpoint_min_silence_ms: int = field(default_factory=lambda: int(os.getenv("ENDPOINT_MIN_SILENCE_MS", "350")))
    endpoint_max_silence_ms: int = field(default_factory=lambda: int(os.getenv("ENDPOINT_MAX_SILENCE_MS", "1500")))
    
    # ===================
    # Speaches API Configuration (Unified STT + TTS)
    # ===================
    speaches_api_url: str = field(default_factory=lambda: os.getenv("SPEACHES_API_URL", "http://localhost:8001"))
    
    # STT Mode: "realtime" (WebSocket streaming) or "batch" (file upload)
    # Realtime streams audio over Speaches' /v1/realtime WebSocket while the
    # caller speaks, so transcription starts the moment the local VAD detects
    # end-of-turn (lower latency than uploading the whole utterance). Requires a
    # Speaches build with the realtime API; falls back to batch automatically if
    # unavailable. Default is batch.
    stt_mode: str = field(default_factory=lambda: os.getenv("STT_MODE", "batch"))

    # Realtime STT: how long to wait for the server transcript after committing
    # the audio buffer (local VAD end-of-turn) before giving up on that turn.
    realtime_commit_timeout_s: float = field(default_factory=lambda: float(os.getenv("REALTIME_COMMIT_TIMEOUT_S", "5.0")))
    
    # STT (Whisper) settings
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-small.en"))
    whisper_language: str = field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "en"))
    whisper_response_format: str = "json"
    
    # API Retry Configuration
    api_retry_attempts: int = field(default_factory=lambda: int(os.getenv("API_RETRY_ATTEMPTS", "3")))
    api_retry_base_delay_s: float = field(default_factory=lambda: float(os.getenv("API_RETRY_BASE_DELAY_S", "0.5")))
    api_retry_max_delay_s: float = field(default_factory=lambda: float(os.getenv("API_RETRY_MAX_DELAY_S", "5.0")))
    api_timeout_s: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_S", "30.0")))
    
    # TTS settings (Piper/Kokoro via Speaches)
    # Default to Kokoro which is well-supported by Speaches
    # Alternative: use piper voices like "rhasspy/piper-voice-en_US-lessac-medium"
    tts_model: str = field(default_factory=lambda: os.getenv("TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX"))
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "af_heart"))
    tts_response_format: str = field(default_factory=lambda: os.getenv("TTS_RESPONSE_FORMAT", "wav"))
    tts_speed: float = field(default_factory=lambda: float(os.getenv("TTS_SPEED", "1.0")))
    # Stream long responses sentence-by-sentence: the first sentence starts
    # playing while the rest are still being synthesized.
    tts_sentence_streaming: bool = field(
        default_factory=lambda: os.getenv("TTS_SENTENCE_STREAMING", "true").lower() == "true")

    # Per-turn acknowledgment before the LLM answer: "chime" plays a short
    # in-memory earcon (instant, no TTS round-trip), "phrase" speaks a random
    # thinking phrase (pre-cached), "none" stays silent.
    turn_ack_mode: str = field(default_factory=lambda: os.getenv("TURN_ACK_MODE", "chime"))
    # Earcon peak amplitude as a fraction of int16 full scale (0.0-1.0].
    chime_volume: float = field(default_factory=lambda: float(os.getenv("CHIME_VOLUME", "0.3")))

    # Deterministic hangup when the caller's whole utterance is a farewell
    # ("bye", "okay thanks, goodbye"): speak a goodbye phrase and end the call
    # without relying on the LLM to invoke HANGUP.
    farewell_hangup_enabled: bool = field(
        default_factory=lambda: os.getenv("FAREWELL_HANGUP_ENABLED", "true").lower() == "true")

    # "Still thinking" earcon: when the LLM response takes longer than the
    # delay, play a soft tick every interval so the caller knows to wait.
    # 0 delay disables it.
    thinking_sound_delay_s: float = field(
        default_factory=lambda: float(os.getenv("THINKING_SOUND_DELAY_S", "2.0")))
    thinking_sound_interval_s: float = field(
        default_factory=lambda: float(os.getenv("THINKING_SOUND_INTERVAL_S", "2.5")))

    # Byte cap for audio uploads to POST /play (decoded and played into the
    # active call).
    play_max_bytes: int = field(
        default_factory=lambda: int(os.getenv("PLAY_AUDIO_MAX_BYTES", str(10 * 1024 * 1024))))

    # Legacy compatibility aliases
    @property
    def whisper_api_url(self) -> str:
        """Alias for backward compatibility."""
        return self.speaches_api_url
    
    @property
    def use_realtime_stt(self) -> bool:
        """Whether to use WebSocket realtime streaming for STT."""
        return self.stt_mode.lower() == "realtime"
    
    # ===================
    # LLM Configuration
    # ===================
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "vllm"))
    # Tool-calling mode: "text" parses [TOOL:...] markers from the response
    # (works with any model); "native" uses the OpenAI tools/function-calling
    # API (requires a backend + model that support it, e.g. recent vLLM).
    llm_tool_calling: str = field(default_factory=lambda: os.getenv("LLM_TOOL_CALLING", "text"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://vllm:8000/v1"))
    # Set-but-empty (e.g. compose's `LLM_API_KEY=${LLM_API_KEY:-}`) falls back
    # to the placeholder too: the OpenAI client refuses an empty api_key, and
    # local OpenAI-compatible backends ignore the value anyway.
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY") or "not-needed")
    
    # Stream LLM tokens straight into sentence-by-sentence TTS: the first
    # sentence starts playing after roughly its own tokens + one TTS call
    # instead of after the full completion. Live-data questions (weather,
    # time, quakes, ...) whose tools are loaded still take the non-streaming
    # path so the grounding retry can replace ungrounded answers. Requires
    # TTS_SENTENCE_STREAMING (streaming exists to speak per sentence; with
    # whole-text TTS requested the turn takes the non-streaming path).
    llm_streaming: bool = field(
        default_factory=lambda: os.getenv("LLM_STREAMING", "true").lower() == "true")

    # Generation
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "512")))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.6")))
    llm_top_p: float = field(default_factory=lambda: float(os.getenv("LLM_TOP_P", "0.85")))
    # Penalize token repetition across the response (OpenAI-compatible
    # frequency_penalty). 0 disables and is omitted from the request entirely,
    # so backends that reject the param are unaffected.
    llm_frequency_penalty: float = field(
        default_factory=lambda: float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0")))
    # Max seconds to wait for the opt-in reformat_for_speech LLM rewrite of an
    # outbound message before falling back to the original text. Generous
    # default: reasoning models (e.g. gpt-oss) think before answering, and the
    # rewrite happens before dialing, not during the call.
    message_reformat_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("MESSAGE_REFORMAT_TIMEOUT_S", "20.0")))

    # Base system prompt (tools + time context are appended at request time by
    # LLMEngine._build_system_prompt). Override precedence:
    #   data/system_prompt.txt (file, highest) > SYSTEM_PROMPT env > built-in default.
    # Set-but-empty env (compose's `SYSTEM_PROMPT=${SYSTEM_PROMPT:-}`) falls
    # through to the default.
    system_prompt: str = field(
        default_factory=lambda: os.getenv("SYSTEM_PROMPT") or _DEFAULT_SYSTEM_PROMPT)
    
    # In-call history window (user+assistant pairs kept verbatim before older
    # turns fold into the rolling summary). Keep this generously large: too
    # small and the agent "forgets" what was said earlier in the same call.
    # 20 turns (~40 messages) sits comfortably inside an 8k-token model context
    # alongside the system prompt, tools, and caller memory.
    max_conversation_turns: int = field(default_factory=lambda: int(os.getenv("MAX_CONVERSATION_TURNS", "20")))

    # Tool-call round trips allowed per turn (native tool-calling loop and the
    # langgraph agent's recursion budget). Bounds how long a model can chain
    # tools on a live phone call.
    llm_max_tool_rounds: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOOL_ROUNDS", "5")))
    # Wall-clock cap on one whole agentic turn (LLM_BACKEND=langgraph): all
    # tool rounds together. On timeout the caller hears an error phrase.
    llm_agent_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("LLM_AGENT_TIMEOUT_S", "30.0")))

    # "Never guess" enforcement: when a turn about live data (weather, time,
    # quakes, GPU, alerts, search) ends with zero tool calls — or the reply
    # merely promises to check — re-run once forcing tool use (grounding.py).
    grounding_retry_enabled: bool = field(
        default_factory=lambda: os.getenv("GROUNDING_RETRY_ENABLED", "true").lower() == "true")
    grounding_retry_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("GROUNDING_RETRY_TIMEOUT_S", "15.0")))

    # ===================
    # Conversation intelligence
    # ===================
    # Cross-call caller memory: remember facts about each caller (keyed by the
    # user part of their SIP URI) between calls, stored under
    # data/caller_memory/. Fail-open: any failure just skips the memory.
    caller_memory_enabled: bool = field(
        default_factory=lambda: os.getenv("CALLER_MEMORY_ENABLED", "true").lower() == "true")
    caller_memory_max_facts: int = field(
        default_factory=lambda: int(os.getenv("CALLER_MEMORY_MAX_FACTS", "15")))
    caller_memory_max_chars: int = field(
        default_factory=lambda: int(os.getenv("CALLER_MEMORY_MAX_CHARS", "1500")))
    # Post-call fact extraction runs off the call path, so this can be generous.
    caller_memory_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("CALLER_MEMORY_TIMEOUT_S", "30.0")))

    # Persona/demeanor tool: let the caller set the agent's speaking style for a
    # call and save/recall named profiles. Profiles persist in persona_file
    # (resolved in __post_init__ to <data_dir>/personas.json).
    enable_persona_tool: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PERSONA_TOOL", "true").lower() == "true")
    persona_file: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("PERSONA_FILE")) if os.getenv("PERSONA_FILE") else None)

    # Knowledge base (RAG): index text/markdown files from knowledge_dir and
    # expose retrieval as the KNOWLEDGE tool. No-ops when the directory is
    # empty or the embedding dependencies are missing.
    knowledge_enabled: bool = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_ENABLED", "true").lower() == "true")
    # Resolved in __post_init__: defaults to <data_dir>/knowledge.
    knowledge_dir: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("KNOWLEDGE_DIR")) if os.getenv("KNOWLEDGE_DIR") else None)
    knowledge_embedding_model: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    knowledge_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "800")))
    knowledge_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "120")))
    knowledge_top_k: int = field(
        default_factory=lambda: int(os.getenv("KNOWLEDGE_TOP_K", "3")))
    # When true, also retrieve against each user utterance and inject the top
    # chunks into the system prompt (zero extra LLM rounds). Default off: the
    # KNOWLEDGE tool lets the model decide when to look something up.
    knowledge_auto_inject: bool = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_AUTO_INJECT", "false").lower() == "true")

    # Rolling conversation summary: when a call outgrows the history window,
    # fold the overflow into a summary block instead of silently dropping it.
    summary_enabled: bool = field(
        default_factory=lambda: os.getenv("CONVERSATION_SUMMARY_ENABLED", "true").lower() == "true")
    summary_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("CONVERSATION_SUMMARY_TIMEOUT_S", "20.0")))


    # ===================
    # Tools
    # ===================
    enable_timer_tool: bool = True
    enable_callback_tool: bool = True
    enable_weather_tool: bool = True
    enable_drink_tool: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DRINK_TOOL", "true").lower() == "true")
    enable_search_tool: bool = False
    enable_calendar_tool: bool = False
    max_timer_duration_hours: int = 24
    callback_retry_attempts: int = 3
    callback_retry_delay_s: int = 60
    callback_ring_timeout_s: int = field(default_factory=lambda: int(os.getenv("CALLBACK_RING_TIMEOUT", "30")))
    
    # Home coordinates for location-aware tools (NWS WEATHER/FORECAST,
    # QUAKES "near"). Empty disables those tools/filters.
    weather_latitude: str = field(default_factory=lambda: os.getenv("WEATHER_LATITUDE", ""))
    weather_longitude: str = field(default_factory=lambda: os.getenv("WEATHER_LONGITUDE", ""))

    # SearxNG instance for the WEB_SEARCH tool (empty disables the tool).
    # The compose files ship an optional service: docker compose --profile search up -d
    searxng_url: str = field(default_factory=lambda: os.getenv("SEARXNG_URL", ""))

    # MCP client: consume tools from external MCP (Model Context Protocol)
    # servers listed in mcp_servers_file (see mcp_tools.load_servers_file for
    # the file format). Disabled by default; fail-open when the file or the
    # optional `mcp` package is missing.
    mcp_enabled: bool = field(
        default_factory=lambda: os.getenv("MCP_ENABLED", "false").lower() == "true")
    # Resolved in __post_init__: defaults to <data_dir>/mcp_servers.json.
    mcp_servers_file: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("MCP_SERVERS_FILE")) if os.getenv("MCP_SERVERS_FILE") else None)
    # Per-tool-call timeout (a phone caller cannot wait 60s); a server entry
    # may override it with its own timeout_s.
    mcp_tool_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("MCP_TOOL_TIMEOUT_S", "10")))
    web_search_max_results: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3")))

    # Observability endpoints for the GPU_STATUS / ALERTS tools. Prometheus is
    # part of docker-compose.observability.yml; tools fail gracefully when
    # it isn't running. ALERTS prefers Alertmanager when configured.
    prometheus_url: str = field(
        default_factory=lambda: os.getenv("PROMETHEUS_URL", "http://prometheus:9090"))
    alertmanager_url: str = field(default_factory=lambda: os.getenv("ALERTMANAGER_URL", ""))

    # CONTAINER_CTL: comma-separated container names the tool may act on.
    # Empty (default) disables the tool entirely. Requires the docker socket
    # mounted into the agent container (see the commented volume in compose).
    container_ctl_allowlist: str = field(
        default_factory=lambda: os.getenv("CONTAINER_CTL_ALLOWLIST", ""))
    docker_socket_path: str = field(
        default_factory=lambda: os.getenv("DOCKER_SOCKET_PATH", "/var/run/docker.sock"))

    # TRANSFER tool (SIP REFER to another extension; same outbound dial policy).
    enable_transfer_tool: bool = field(
        default_factory=lambda: os.getenv("ENABLE_TRANSFER_TOOL", "true").lower() == "true")
    
    # ===================
    # REST API / Webhook security & limits
    # ===================
    # If set, all mutating endpoints require this token via
    # `Authorization: Bearer <token>` or `X-API-Key: <token>`. Empty = open (dev only).
    api_auth_token: str = field(default_factory=lambda: os.getenv("API_AUTH_TOKEN", ""))
    # Interface the REST API binds to. Defaults to loopback so a fresh install is
    # not exposed; set API_HOST=0.0.0.0 to listen on all interfaces (e.g. in
    # Docker) — but then API_AUTH_TOKEN must be set, or the agent refuses to start
    # (override with ALLOW_UNAUTHENTICATED=true).
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    # Host-side address the API port is actually published on (set by the
    # compose files from API_BIND_ADDRESS). In Docker the container must bind
    # 0.0.0.0 for the published port to work, so the startup exposure check
    # uses this value, when set, instead of api_host. Empty = fall back to
    # api_host (direct/non-Docker runs).
    api_published_host: str = field(default_factory=lambda: os.getenv("API_BIND_ADDRESS", ""))
    # Auto-discover extra tools from plugins/ directories and data/plugins
    # (mounted volume) in addition to the explicitly registered builtins.
    enable_plugin_autodiscovery: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PLUGIN_AUTODISCOVERY", "true").lower() == "true")

    # Answering-machine detection for outbound notification calls (heuristic:
    # a human answers briefly then waits; a machine keeps talking). When
    # enabled, sustained speech right after answer marks the call
    # machine_answered=true in the webhook payload. Off by default.
    amd_enabled: bool = field(
        default_factory=lambda: os.getenv("AMD_ENABLED", "false").lower() == "true")
    # How long to listen after answer before playing the message.
    amd_window_s: float = field(default_factory=lambda: float(os.getenv("AMD_WINDOW_S", "2.5")))
    # Continuous speech longer than this classifies the answerer as a machine.
    amd_machine_speech_ms: int = field(
        default_factory=lambda: int(os.getenv("AMD_MACHINE_SPEECH_MS", "1500")))

    # Rate limit for mutating endpoints: requests per minute per client (keyed
    # by API credential, else client IP). 0 disables rate limiting.
    rate_limit_rpm: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_RPM", "0")))
    # Token-bucket burst size; defaults to the per-minute rate when 0.
    rate_limit_burst: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_BURST", "0")))
    # Escape hatch: permit an externally-bound API with no auth token. Off by
    # default so the insecure combination fails closed at startup.
    allow_unauthenticated: bool = field(
        default_factory=lambda: os.getenv("ALLOW_UNAUTHENTICATED", "false").lower() == "true")

    # Operator dashboard: GET /admin serves a self-contained static page (the
    # data endpoints it calls are auth-gated separately). Set false to remove
    # the page (404) entirely.
    admin_ui_enabled: bool = field(
        default_factory=lambda: os.getenv("ADMIN_UI_ENABLED", "true").lower() == "true")

    # SSRF guard for callback_url webhooks. When False, callback URLs that resolve
    # to loopback/private/link-local/reserved addresses are rejected.
    webhook_allow_private: bool = field(
        default_factory=lambda: os.getenv("WEBHOOK_ALLOW_PRIVATE", "false").lower() == "true")
    webhook_require_https: bool = field(
        default_factory=lambda: os.getenv("WEBHOOK_REQUIRE_HTTPS", "false").lower() == "true")
    # Optional HMAC-SHA256 secret for outgoing webhooks. When set, each webhook
    # POST carries X-Timestamp plus X-Signature: sha256=<HMAC(secret,
    # "<timestamp>.<body>")> so receivers can verify authenticity and freshness.
    webhook_signing_secret: str = field(
        default_factory=lambda: os.getenv("WEBHOOK_SIGNING_SECRET", ""))

    # Call-lifecycle event webhook. When set, the agent POSTs signed
    # call.started / call.ended events here for every call (inbound and
    # outbound). Empty (the default) disables the feature. Private URLs
    # (e.g. http://n8n:5678/...) additionally require WEBHOOK_ALLOW_PRIVATE=true.
    call_event_webhook_url: str = field(
        default_factory=lambda: os.getenv("CALL_EVENT_WEBHOOK_URL", ""))
    # Comma-separated subset of events to emit.
    call_events: str = field(
        default_factory=lambda: os.getenv("CALL_EVENTS", "call.started,call.ended"))
    # Include the finished transcript in call.ended payloads.
    call_event_include_transcript: bool = field(
        default_factory=lambda: os.getenv("CALL_EVENT_INCLUDE_TRANSCRIPT", "true").lower() == "true")

    # One live call at a time: decline a second inbound INVITE with 486 Busy
    # Here instead of evicting the current caller mid-conversation. Set false
    # to restore the old replace-the-call behavior.
    sip_busy_reject: bool = field(
        default_factory=lambda: os.getenv("SIP_BUSY_REJECT", "true").lower() == "true")

    # Maximum simultaneous live calls. The default of 1 keeps the shipped
    # single-call behavior; raising it lets additional inbound INVITEs run
    # their own concurrent sessions instead of being 486-rejected. 2-3 is
    # realistic on a DGX Spark — every concurrent call contends for the same
    # STT/TTS/LLM services, so latency stacks with each added call.
    max_concurrent_calls: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_CALLS", "1")))

    # PJSIP call-slot count (pjsua uaConfig.maxCalls) — the STACK-level cap.
    # Outbound calls and just-rejected INVITEs briefly hold slots too, so it
    # must exceed MAX_CONCURRENT_CALLS with headroom or pjsua 486-rejects
    # INVITEs before the app's capacity gate ever runs; sip_handler raises it
    # to MAX_CONCURRENT_CALLS + 2 (with a warning) when set lower.
    sip_max_calls: int = field(
        default_factory=lambda: int(os.getenv("SIP_MAX_CALLS", "4")))

    # IANA timezone for everything spoken or scheduled in local wall-clock
    # terms: the system-prompt clock, the DATETIME tool default, and the
    # recurring-schedule comparisons. The container itself may run UTC.
    local_timezone: str = field(
        default_factory=lambda: os.getenv("LOCAL_TIMEZONE", "America/Los_Angeles"))

    def local_now(self) -> datetime:
        """Naive wall-clock in local_timezone — THE clock for scheduled-task
        times. Everything that writes or compares ScheduledTask.execute_at
        must use this (scheduler, /schedule endpoints, STATUS tool) so
        remaining-time math never mixes clocks. Falls back to the system
        clock when the zone name is invalid.
        """
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(self.local_timezone)).replace(tzinfo=None)
        except Exception:
            return datetime.now()

    # ===================
    # Virtual numbers (ephemeral inbound extensions)
    # ===================
    # Single-use temporary extensions created via POST /virtual-numbers: the
    # agent answers a call dialed to one with per-number context, webhooks the
    # outcome, and clears the number. Disabled by default.
    virtual_numbers_enabled: bool = field(
        default_factory=lambda: os.getenv("VIRTUAL_NUMBERS_ENABLED", "false").lower() == "true")
    virtual_number_default_ttl_s: int = field(
        default_factory=lambda: int(os.getenv("VIRTUAL_NUMBER_DEFAULT_TTL_S", "900")))
    virtual_number_max_ttl_s: int = field(
        default_factory=lambda: int(os.getenv("VIRTUAL_NUMBER_MAX_TTL_S", "86400")))
    # Inclusive numeric range for auto-allocated extensions ("start-end").
    virtual_number_range: str = field(
        default_factory=lambda: os.getenv("VIRTUAL_NUMBER_RANGE", "7300-7399"))
    virtual_number_max_active: int = field(
        default_factory=lambda: int(os.getenv("VIRTUAL_NUMBER_MAX_ACTIVE", "50")))

    # Outbound dial-target policy. When False, callers may not supply a raw
    # `sip:` URI or an `@domain` part in `extension` (prevents routing calls to
    # arbitrary external SIP domains); the target is always built as
    # `sip:<extension>@<sip_domain>`. Optional regex further restricts extensions.
    outbound_allow_sip_uri: bool = field(
        default_factory=lambda: os.getenv("OUTBOUND_ALLOW_SIP_URI", "false").lower() == "true")
    outbound_extension_pattern: str = field(
        default_factory=lambda: os.getenv("OUTBOUND_EXTENSION_PATTERN", ""))

    # Bounds to prevent a single request from monopolising the call pipeline.
    max_ring_timeout_s: int = field(default_factory=lambda: int(os.getenv("MAX_RING_TIMEOUT_S", "120")))
    max_choice_timeout_s: int = field(default_factory=lambda: int(os.getenv("MAX_CHOICE_TIMEOUT_S", "120")))
    max_choice_repeat: int = field(default_factory=lambda: int(os.getenv("MAX_CHOICE_REPEAT", "5")))
    # Backpressure: reject new calls past these caps (HTTP 429).
    max_queue_depth: int = field(default_factory=lambda: int(os.getenv("MAX_QUEUE_DEPTH", "1000")))
    max_direct_concurrent_calls: int = field(
        default_factory=lambda: int(os.getenv("MAX_DIRECT_CONCURRENT_CALLS", "20")))

    # ===================
    # Phrases Configuration
    # ===================
    phrases: PhrasesConfig = field(default_factory=PhrasesConfig)
    
    # ===================
    # System
    # ===================
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "recordings").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        (self.data_dir / "caller_memory").mkdir(exist_ok=True)

        # Knowledge dir defaults relative to data_dir, which isn't known until
        # the dataclass is built.
        if self.knowledge_dir is None:
            self.knowledge_dir = self.data_dir / "knowledge"

        # MCP servers file defaults relative to data_dir too.
        if self.mcp_servers_file is None:
            self.mcp_servers_file = self.data_dir / "mcp_servers.json"

        # Persona profiles file defaults relative to data_dir too.
        if self.persona_file is None:
            self.persona_file = self.data_dir / "personas.json"
        
        # Load phrases from JSON file if it exists
        phrases_file = self.data_dir / "phrases.json"
        if phrases_file.exists():
            self._load_phrases_from_file(phrases_file)

        # System prompt file override (wins over SYSTEM_PROMPT env, mirroring
        # the phrases.json precedence). Live-editable via the ./data mount.
        prompt_file = self.data_dir / "system_prompt.txt"
        if prompt_file.exists():
            try:
                text = prompt_file.read_text().strip()
                if text:
                    self.system_prompt = text
                else:
                    print(f"Warning: {prompt_file} is empty, keeping current system prompt")
            except Exception as e:
                print(f"Warning: Could not load system prompt from {prompt_file}: {e}")

        # Validate the turn-ack mode; fall back rather than crash the agent.
        mode = self.turn_ack_mode.lower()
        if mode not in ("chime", "phrase", "none"):
            print(f"Warning: invalid TURN_ACK_MODE '{self.turn_ack_mode}', falling back to 'chime'")
            mode = "chime"
        self.turn_ack_mode = mode
        self.chime_volume = min(max(self.chime_volume, 0.01), 1.0)

        # Validate the endpointing mode the same way; fall back to the
        # zero-behavior-change default rather than crash the agent.
        endpoint_mode = self.endpoint_mode.lower()
        if endpoint_mode not in ("fixed", "adaptive", "speculative"):
            print(f"Warning: invalid ENDPOINT_MODE '{self.endpoint_mode}', falling back to 'fixed'")
            endpoint_mode = "fixed"
        self.endpoint_mode = endpoint_mode

    def _load_phrases_from_file(self, filepath: Path):
        """Load phrases from a JSON file."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Only assign values that are actually lists; a stray scalar would
            # otherwise crash get_all_phrases_for_cache (str + list) at startup.
            for key in (
                "greetings",
                "goodbyes",
                "thinking",
                "errors",
                "followups",
                "precache_extra",
            ):
                if key not in data:
                    continue
                value = data[key]
                if not isinstance(value, list):
                    print(f"Warning: Ignoring '{key}' in {filepath}: expected a list, got {type(value).__name__}")
                    continue
                setattr(self.phrases, key, value)

        except Exception as e:
            print(f"Warning: Could not load phrases from {filepath}: {e}")


# Singleton
_config: Optional[Config] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
