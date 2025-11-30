---
title: "Configuration"
excerpt: "Complete configuration reference"
category: "Setup"
slug: "configuration"
---

# Configuration Reference

All configuration is done via environment variables. This page documents every available option.

## SIP Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIP_USERNAME` | Yes | - | SIP account username |
| `SIP_PASSWORD` | Yes | - | SIP account password |
| `SIP_DOMAIN` | Yes | - | SIP server domain/IP |
| `SIP_PORT` | No | `5060` | SIP server port |
| `SIP_TRANSPORT` | No | `udp` | Transport protocol (udp/tcp/tls) |
| `SIP_DISPLAY_NAME` | No | `AI Assistant` | Caller ID display name |

## LLM Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_BASE_URL` | Yes | - | OpenAI-compatible API base URL |
| `LLM_MODEL` | Yes | - | Model name to use |
| `LLM_API_KEY` | No | `not-needed` | API key (if required) |
| `LLM_MAX_TOKENS` | No | `150` | Maximum response tokens |
| `LLM_TEMPERATURE` | No | `0.7` | Response creativity (0.0-1.0) |
| `LLM_TIMEOUT` | No | `30` | Request timeout in seconds |

## Speech-to-Text (STT) Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STT_BASE_URL` | Yes | - | Whisper API base URL |
| `STT_MODEL` | No | `whisper-1` | STT model name |
| `STT_LANGUAGE` | No | `en` | Language code |
| `STT_MODE` | No | `batch` | Mode: `batch` or `realtime` |

### STT Modes

- **batch** (recommended): Audio is buffered locally, sent when silence detected. More stable.
- **realtime**: Audio streams continuously to server. Requires compatible server (experimental).

## Text-to-Speech (TTS) Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TTS_BASE_URL` | Yes | - | TTS API base URL |
| `TTS_VOICE` | No | `default` | Voice name/ID |
| `TTS_SPEED` | No | `1.0` | Speech speed multiplier |
| `TTS_PROVIDER` | No | `xtts` | Provider: `xtts`, `piper`, `fish`, `openai` |

### TTS Provider Examples

**XTTS:**
```env
TTS_BASE_URL=http://xtts-server:8000
TTS_PROVIDER=xtts
TTS_VOICE=default
```

**Piper:**
```env
TTS_BASE_URL=http://piper-server:5000
TTS_PROVIDER=piper
TTS_VOICE=en_US-lessac-medium
```

**Fish Speech:**
```env
TTS_BASE_URL=http://fish-server:8080
TTS_PROVIDER=fish
TTS_VOICE=default
```

## Assistant Behavior

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ASSISTANT_NAME` | No | `Assistant` | Name the assistant uses |
| `COMPANY_NAME` | No | `CHAOS.CORP` | Company name in greetings |
| `GREETING_MESSAGE` | No | (built-in) | Custom greeting message |
| `SYSTEM_PROMPT` | No | (built-in) | Custom system prompt |
| `MAX_CONVERSATION_TURNS` | No | `50` | Max turns before call timeout |
| `CALL_TIMEOUT_SECONDS` | No | `300` | Max call duration (5 min) |
| `SILENCE_TIMEOUT_SECONDS` | No | `10` | Hangup after silence |

## Tool Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_TIMER_TOOL` | No | `true` | Enable SET_TIMER tool |
| `ENABLE_CALLBACK_TOOL` | No | `true` | Enable CALLBACK tool |
| `ENABLE_WEATHER_TOOL` | No | `true` | Enable WEATHER tool |
| `MAX_TIMER_DURATION_HOURS` | No | `24` | Maximum timer length |
| `CALLBACK_RETRY_ATTEMPTS` | No | `3` | Retries for failed callbacks |
| `CALLBACK_RETRY_DELAY_S` | No | `30` | Delay between retries |

## Weather (Tempest) Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TEMPEST_STATION_ID` | No | - | WeatherFlow station ID |
| `TEMPEST_API_TOKEN` | No | - | WeatherFlow API token |

Get your station ID and token from [tempestwx.com](https://tempestwx.com/).

## API Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_PORT` | No | `8080` | REST API listen port |
| `API_HOST` | No | `0.0.0.0` | REST API listen address |
| `MAX_CONCURRENT_CALLS` | No | `1` | Max simultaneous outbound calls |

## Telemetry Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `METRICS_PORT` | No | `9090` | Prometheus metrics port |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | - | OpenTelemetry collector endpoint |
| `OTEL_SERVICE_NAME` | No | `sip-agent` | Service name for traces |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `LOG_FORMAT` | No | `json` | Log format: `json` or `text` |

## Audio Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AUDIO_SAMPLE_RATE` | No | `16000` | Audio sample rate (Hz) |
| `VAD_THRESHOLD` | No | `0.5` | Voice activity detection threshold |
| `VAD_MIN_SILENCE_MS` | No | `500` | Silence before end of speech |
| `BARGE_IN_ENABLED` | No | `true` | Allow interrupting assistant |

## Complete Example

```env
# =============================================================================
# SIP AI Assistant Configuration
# =============================================================================

# SIP Connection
SIP_USERNAME=assistant
SIP_PASSWORD=secretpassword123
SIP_DOMAIN=pbx.example.com
SIP_PORT=5060
SIP_DISPLAY_NAME=AI Assistant

# LLM (Language Model)
LLM_BASE_URL=http://vllm-server:8000/v1
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_MAX_TOKENS=200
LLM_TEMPERATURE=0.7

# Speech-to-Text (Whisper)
STT_BASE_URL=http://speaches:8000/v1
STT_MODEL=whisper-1
STT_MODE=batch

# Text-to-Speech
TTS_BASE_URL=http://xtts:8000
TTS_PROVIDER=xtts
TTS_VOICE=default

# Assistant Behavior
ASSISTANT_NAME=Friday
COMPANY_NAME=Stark Industries
CALL_TIMEOUT_SECONDS=600

# Tools
ENABLE_TIMER_TOOL=true
ENABLE_CALLBACK_TOOL=true
ENABLE_WEATHER_TOOL=true
MAX_TIMER_DURATION_HOURS=24

# Weather Station
TEMPEST_STATION_ID=12345
TEMPEST_API_TOKEN=your-api-token

# API
API_PORT=8080
MAX_CONCURRENT_CALLS=2

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Environment Variable Precedence

1. Environment variables set directly
2. `.env` file in the working directory
3. Default values

## Secrets Management

For production, consider using Docker secrets or a secrets manager:

```yaml
# docker-compose.yml with secrets
services:
  sip-agent:
    secrets:
      - sip_password
      - llm_api_key
    environment:
      - SIP_PASSWORD_FILE=/run/secrets/sip_password

secrets:
  sip_password:
    file: ./secrets/sip_password.txt
```
