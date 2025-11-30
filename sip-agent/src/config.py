"""
Low-Latency Configuration with Speaches (Unified STT + TTS)
============================================================
All ML inference offloaded to dedicated API services:
- Speaches API for both STT (Whisper) and TTS (Piper/Kokoro)
- vLLM for LLM
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


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
    
    # Speech detection
    speech_pad_ms: int = 200
    min_speech_duration_ms: int = field(default_factory=lambda: int(os.getenv("MIN_SPEECH_DURATION_MS", "200")))
    max_speech_duration_s: float = field(default_factory=lambda: float(os.getenv("MAX_SPEECH_DURATION_S", "10.0")))
    silence_duration_ms: int = field(default_factory=lambda: int(os.getenv("SILENCE_TIMEOUT_MS", "750")))
    
    # ===================
    # Speaches API Configuration (Unified STT + TTS)
    # ===================
    speaches_api_url: str = field(default_factory=lambda: os.getenv("SPEACHES_API_URL", "http://localhost:8001"))
    
    # STT Mode: "realtime" (WebSocket streaming) or "batch" (file upload)
    # NOTE: Realtime mode requires Speaches v0.8.0+ with stable realtime API.
    # Default to batch mode for stability. Set STT_MODE=realtime to enable streaming.
    stt_mode: str = field(default_factory=lambda: os.getenv("STT_MODE", "batch"))
    
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
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://vllm:8000/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "not-needed"))
    
    # Generation
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "512")))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.6")))
    llm_top_p: float = field(default_factory=lambda: float(os.getenv("LLM_TOP_P", "0.85")))
    
    max_conversation_turns: int = field(default_factory=lambda: int(os.getenv("MAX_CONVERSATION_TURNS", "10")))
    
    # ===================
    # Tools
    # ===================
    enable_timer_tool: bool = True
    enable_callback_tool: bool = True
    enable_weather_tool: bool = True
    enable_search_tool: bool = False
    enable_calendar_tool: bool = False
    max_timer_duration_hours: int = 24
    callback_retry_attempts: int = 3
    callback_retry_delay_s: int = 60
    callback_ring_timeout_s: int = field(default_factory=lambda: int(os.getenv("CALLBACK_RING_TIMEOUT", "30")))
    
    # Tempest Weather API
    tempest_station_id: str = field(default_factory=lambda: os.getenv("TEMPEST_STATION_ID", ""))
    tempest_api_token: str = field(default_factory=lambda: os.getenv("TEMPEST_API_TOKEN", ""))
    
    # ===================
    # System
    # ===================
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "recordings").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        
    @property
    def system_prompt(self) -> str:
        """
        Base system prompt (without tools section).
        
        Tools are dynamically added by ToolManager.get_tools_prompt()
        """
        return """You are a voice assistant  on a phone call. Follow these guidelines:

RULES:
- Keep responses SHORT (2-6 sentences max)
- Be conversational, not formal
- No markdown or formatting
- If confused, ask briefly

Be helpful and concise."""


# Singleton
_config: Optional[Config] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
