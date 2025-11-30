# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-30

### Added

#### Core Features
- SIP client with full call handling (incoming/outgoing)
- RTP audio streaming with codec negotiation (PCMU, PCMA, Opus)
- Voice Activity Detection (VAD) with configurable aggressiveness
- Barge-in support for natural conversation flow
- Conversation history tracking per call

#### Speech Processing
- Speech-to-text via Speaches API (Whisper backend)
- Text-to-speech via Speaches API (Kokoro backend)
- Batch STT mode for stability
- Experimental WebSocket realtime STT mode
- Audio pre-caching for common phrases
- Configurable voice and speed settings

#### LLM Integration
- OpenAI-compatible API support
- vLLM backend support
- Ollama backend support
- LM Studio compatibility
- Configurable temperature, top_p, max_tokens
- Tool calling via `[TOOL:NAME:params]` syntax

#### Built-in Tools
- `WEATHER` - Tempest weather station integration
- `SET_TIMER` - Countdown timers with voice alerts
- `CALLBACK` - Scheduled callbacks
- `HANGUP` - Graceful call termination
- `STATUS` - Pending task status
- `CANCEL` - Cancel timers/callbacks
- `DATETIME` - Current date and time
- `CALC` - Math calculations
- `JOKE` - Random jokes (general, tech, dad categories)
- `SIMON_SAYS` - Echo back verbatim

#### REST API
- `GET /health` - Health check endpoint
- `POST /call` - Initiate outbound calls
- `GET /call/{id}` - Get call status
- `DELETE /call/{id}` - Hang up call
- `GET /tools` - List available tools
- `POST /tools/{name}/call` - Execute tool via call
- `POST /schedule` - Schedule calls
- `GET /schedule` - List scheduled calls
- `DELETE /schedule/{id}` - Cancel scheduled call

#### Scheduled Calls
- One-time scheduled calls
- Recurring calls (daily, weekdays, weekends)
- Tool execution on schedule
- Custom message prefixes

#### Configuration
- Environment variable configuration
- Customizable phrases via env vars or JSON file
- Phrase categories: greetings, goodbyes, acknowledgments, thinking, errors, followups
- Pre-cache configuration for TTS optimization

#### Plugin System
- `BaseTool` abstract class for custom tools
- Parameter validation
- Async execution support
- Access to call context and resources
- Hot-reload support (planned)

#### Observability
- Prometheus metrics endpoint
- OpenTelemetry tracing support
- Structured JSON logging
- Grafana dashboard template
- Log viewer utility script

#### Documentation
- Comprehensive README with Mermaid diagrams
- Getting Started guide
- Configuration reference (40+ variables)
- API reference with curl examples
- Tools documentation
- Plugin development guide
- Integration examples (Home Assistant, n8n, Grafana, cron)
- readme.io compatible formatting

#### DevOps
- Docker and Docker Compose support
- Multi-architecture builds (ARM64, x86_64)
- GitHub Actions workflow for readme.io sync
- Health check endpoints
- Graceful shutdown handling

### Technical Details

#### Dependencies
- Python 3.11+
- FastAPI for REST API
- PJSIP for SIP handling
- httpx for async HTTP
- webrtcvad for voice activity detection
- numpy for audio processing
- Redis for call queue (optional)

#### Supported Platforms
- NVIDIA DGX Spark (primary target)
- Any Linux with Docker support
- GPU recommended for local LLM/STT/TTS

### Notes

- This is the initial public release
- WebSocket realtime STT is experimental
- Default LLM model is `openai-community/gpt2-xl`
- Requires external Speaches server for STT/TTS
- Requires external LLM server (vLLM, Ollama, etc.)

---

## [Unreleased]

### Planned
- Music on hold
- Call transfer
- Calendar integration
- Web search tool
- Home Assistant native integration
- SMS notifications
- Multi-language support
- Voice cloning support
- Call recording and transcription export
