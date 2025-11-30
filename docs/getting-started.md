---
title: "Getting Started"
excerpt: "Install and configure the SIP AI Assistant"
category: "Setup"
slug: "getting-started"
---

# Getting Started

This guide walks you through setting up the SIP AI Assistant.

## Prerequisites

- Docker and Docker Compose
- A SIP server (FreePBX, Asterisk, or any SIP-compatible PBX)
- An LLM server (vLLM, LM Studio, OpenAI API, etc.)
- A Whisper-compatible STT server (Speaches, Whisper.cpp, etc.)
- A TTS server (XTTS, Piper, Fish Speech, etc.)

## Quick Start with Docker Compose

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/sip-agent.git
cd sip-agent
```

### 2. Configure Environment

Copy the example environment file and edit it:

```bash
cp .env.example .env
nano .env
```

Minimum required configuration:

```env
# SIP Settings
SIP_USERNAME=assistant
SIP_PASSWORD=your-password
SIP_DOMAIN=your-pbx.local

# LLM Settings
LLM_BASE_URL=http://your-llm-server:8000/v1
LLM_MODEL=your-model-name

# STT Settings (Whisper)
STT_BASE_URL=http://your-whisper-server:8000/v1

# TTS Settings
TTS_BASE_URL=http://your-tts-server:8000
TTS_VOICE=default
```

### 3. Start the Service

```bash
docker compose up -d
```

### 4. Verify It's Running

Check the health endpoint:

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{
  "status": "healthy",
  "sip_registered": true,
  "queue": {
    "pending": 0,
    "active": 0,
    "max_concurrent": 1
  }
}
```

### 5. Make a Test Call

From your SIP phone, dial the extension assigned to the assistant. You should hear a greeting and can start talking!

## Docker Compose Configuration

Here's a complete `docker-compose.yml` example:

```yaml
version: '3.8'

services:
  sip-agent:
    image: sip-agent:latest
    build: ./sip-agent
    container_name: sip-agent
    ports:
      - "8080:8080"      # API
      - "5060:5060/udp"  # SIP
      - "10000-10100:10000-10100/udp"  # RTP
    environment:
      - SIP_USERNAME=${SIP_USERNAME}
      - SIP_PASSWORD=${SIP_PASSWORD}
      - SIP_DOMAIN=${SIP_DOMAIN}
      - LLM_BASE_URL=${LLM_BASE_URL}
      - LLM_MODEL=${LLM_MODEL}
      - STT_BASE_URL=${STT_BASE_URL}
      - TTS_BASE_URL=${TTS_BASE_URL}
    volumes:
      - ./plugins:/app/plugins  # Custom plugins
    restart: unless-stopped

  # Optional: Prometheus metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Directory Structure

```
sip-agent/
├── src/
│   ├── main.py           # Application entry point
│   ├── config.py         # Configuration management
│   ├── api.py            # REST API endpoints
│   ├── sip_handler.py    # SIP protocol handling
│   ├── audio_pipeline.py # STT/TTS processing
│   ├── llm_engine.py     # LLM integration
│   ├── tool_manager.py   # Tool execution
│   └── plugins/          # Built-in tool plugins
│       ├── weather_tool.py
│       ├── timer_tool.py
│       ├── callback_tool.py
│       └── ...
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
```

## Verifying Components

### Check SIP Registration

```bash
curl http://localhost:8080/health | jq .sip_registered
```

### List Available Tools

```bash
curl http://localhost:8080/tools | jq
```

### Test a Tool

```bash
curl -X POST http://localhost:8080/tools/DATETIME/execute \
  -H "Content-Type: application/json" \
  -d '{"params": {"format": "full"}}'
```

## Troubleshooting

### SIP Not Registering

1. Check your PBX allows the IP address
2. Verify credentials in `.env`
3. Check firewall allows UDP 5060

```bash
# View SIP logs
docker logs sip-agent 2>&1 | grep -i sip
```

### No Audio

1. Verify RTP port range is open (10000-10100/udp)
2. Check TTS server is responding
3. Verify STT server is processing audio

### LLM Not Responding

1. Test LLM endpoint directly:
   ```bash
   curl http://your-llm-server:8000/v1/models
   ```
2. Check `LLM_MODEL` matches available models

## Next Steps

- [Configuration Reference](configuration) - Full list of options
- [API Documentation](api-reference) - REST API details
- [Available Tools](tools) - Built-in capabilities
- [Creating Plugins](plugins) - Add custom tools
