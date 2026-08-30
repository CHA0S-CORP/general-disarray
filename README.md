# 📞⚡ General Dissarray
## 🤖 SIP Enabled AI Agent

> 🤖 **ROBO CODED** — This project was made with AI and may not be 100% sane. But the code does work! 🎉

A voice-powered AI assistant that answers phone calls, understands natural language, and performs actions like checking weather, setting timers, scheduling callbacks, and more.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/Version-0.1.0-green.svg)](RELEASE.md)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Runs on DGX Spark](https://img.shields.io/badge/Runs%20on-DGX%20Spark-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Docs](https://img.shields.io/badge/Docs-readme.io-blue)](https://sip-agent.readme.io)

[![Build Status](https://github.com/CHA0S-CORP/general-disarray/actions/workflows/docker-build.yml/badge.svg)](https://github.com/CHA0S-CORP/general-disarray/actions/workflows/docker-build.yml)
[![Build Status](https://github.com/CHA0S-CORP/general-disarray/actions/workflows/docker-build-nvitop_exporter.yml/badge.svg)](https://github.com/CHA0S-CORP/general-disarray/actions/workflows/docker-build-nvitop_exporter.yml)

📖 **[Read the Documentation](https://sip-agent.readme.io)**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Voice Conversations** | Natural speech-to-text and text-to-speech powered by Whisper & Kokoro |
| 🤖 **LLM Integration** | Connects to OpenAI, vLLM, Ollama, LM Studio, and more |
| 🔧 **Built-in Tools** | Weather, timers, callbacks, date/time, calculator, jokes |
| 🔌 **Plugin System** | Easily add custom tools with Python |
| 🌐 **REST API** | Initiate outbound calls, execute tools, manage schedules |
| ⏰ **Scheduled Calls** | One-time or recurring calls (daily briefings, reminders) |
| 🔗 **Webhooks** | Trigger calls from Home Assistant, n8n, Grafana, and more |
| 🗣️ **Custom Phrases** | Customize greetings, goodbyes, and responses via JSON or env vars |
| 📊 **Observability** | Prometheus metrics, OpenTelemetry tracing, structured JSON logs |

## 💡 Use Cases

| Use Case | Example |
|----------|---------|
| ⏲️ **Timers & Reminders** | *"Set a timer for 10 minutes"* |
| 📞 **Callbacks** | *"Call me back in an hour"* |
| 🌤️ **Weather Briefings** | Scheduled morning weather calls |
| 📅 **Appointment Reminders** | Outbound calls with confirmation |
| 🚨 **Alerts & Notifications** | Webhook-triggered phone calls |
| 🏠 **Smart Home** | Voice control via phone |

---
## 🚀 Quick Example

Call the assistant and say:

> 🗣️ *"What's the weather like?"*

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Agent as 🤖 SIP Agent
    participant STT as 🎤 Speaches
    participant LLM as 🧠 LLM
    participant Tool as 🌤️ Weather Tool
    
    User->>Agent: "What's the weather like?"
    Agent->>STT: Audio stream
    STT-->>Agent: Transcribed text
    Agent->>LLM: User query + context
    LLM-->>Agent: [TOOL:WEATHER]
    Agent->>Tool: Execute
    Tool-->>Agent: Weather data
    Agent->>LLM: Tool result
    LLM-->>Agent: Natural response
    Agent->>STT: Text to speech
    STT-->>Agent: Audio
    Agent->>User: "At Storm Lake, it's 44°..."
```

**Assistant responds:**

> 🤖 *"At Storm Lake, as of 9:30 pm, it's 44 degrees with foggy conditions. Wind is calm."*

![Example conversation flow](https://github.com/MaxwellDPS/docs/blob/v1.0/photos/Screenshot%202025-11-29%20214959.png?raw=true)
<!-- TODO: Screenshot of log viewer showing a weather query conversation -->
---

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Caller
        Phone[📱 SIP Phone]
    end
    
    subgraph Agent["🤖 SIP AI Agent"]
        SIP[SIP Client]
        Audio[Audio Pipeline]
        Tools[Tool Manager]
        API[REST API]
    end
    
    subgraph Services
        LLM[🧠 LLM Server<br/>OpenAI / vLLM / Ollama]
        Speaches[🎤 Speaches<br/>STT + TTS]
    end
    
    subgraph Integrations
        HA[🏠 Home Assistant]
        N8N[🔄 n8n]
        Webhook[🔗 Webhooks]
    end
    
    Phone <-->|SIP/RTP| SIP
    SIP <--> Audio
    Audio <-->|Whisper| Speaches
    Audio <-->|Kokoro| Speaches
    Audio <--> Tools
    Tools <-->|OpenAI API| LLM
    
    API <--> Tools
    HA -->|HTTP| API
    N8N -->|HTTP| API
    Webhook -->|HTTP| API
```
---

## 🔗 Services & Integrations

| Service | Purpose | URL |
|---------|---------|-----|
| 🤖 **SIP Agent** | AI Voice Assistant API | [localhost:8080](http://localhost:8080) |
| 🎤 **Speaches** | STT/TTS (Whisper + Kokoro) | [localhost:8001](http://localhost:8001) |
| 🧠 **vLLM** | LLM Inference | [localhost:8000](http://localhost:8000) |
| 🔴 **Redis** | Call Queue & Cache | `redis://localhost:6379` |
| 📊 **Prometheus** | Metrics Collection | [localhost:9090](http://localhost:9090) |
| 📈 **Grafana** | Dashboards | [localhost:3000](http://localhost:3000) |
| 📝 **Loki** | Log Aggregation | [localhost:3100](http://localhost:3100) |
| 🔍 **Tempo** | Distributed Tracing | [localhost:3200](http://localhost:3200) |
| 🔄 **n8n** | Workflow Automation | [localhost:5678](http://localhost:5678) |
---

## 🚀 Quick Start

### Prerequisites

| Requirement | Description |
|-------------|-------------|
| 🐳 **Docker** | Docker and Docker Compose |
| 📞 **SIP Server** | FreePBX, Asterisk, 3CX, or any SIP PBX |

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/sip-agent.git
cd sip-agent

# Configure environment
cp sip-agent/.env.example sip-agent/.env
nano sip-agent/.env

# Start services
docker compose up -d

# (Optional) Start services with Observability
docker compose -f ./docker-compose.yml -f docker-compose.observability.yml up -d
```

### Verify Installation

```bash
curl http://localhost:8080/health | jq
```

**Expected output:**

```json
{
  "status": "healthy",
  "sip_registered": true,
  "active_calls": 0
}
```

### Make a Test Call

```
┌────────────────────────────────────────────────────────────┐
│ 📞 INCOMING CALL                                           │
├────────────────────────────────────────────────────────────┤
│ 🤖 "Hello! Welcome to the AI assistant. How can I help?"  │
│ 👤 "What's the weather like?"                              │
│ 🤖 "At Storm Lake, it's 44 degrees with foggy conditions."│
│ 👤 "Set a timer for 5 minutes"                             │
│ 🤖 "Timer set for 5 minutes!"                             │
│ 👤 "Goodbye"                                               │
│ 🤖 "Goodbye! Have a great day!"                           │
└────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

Create a `.env` file with your settings:

```env
# 📞 SIP Connection
SIP_USER=ai-assistant
SIP_PASSWORD=your-secure-password
SIP_DOMAIN=pbx.example.com

# 🎤 Speaches (STT + TTS)
SPEACHES_API_URL=http://speaches:8001

# 🧠 LLM Settings
LLM_BASE_URL=http://vllm:8000/v1
LLM_MODEL=openai-community/gpt2-xl

# 🌤️ Weather (Optional)
TEMPEST_STATION_ID=12345
TEMPEST_API_TOKEN=your-api-token
```

📖 See [Configuration Reference](https://sip-agent.readme.io/docs/configuration) for all options.

---

## 🌐 API Examples

### 📞 Make an Outbound Call

```bash
curl -X POST http://localhost:8080/call \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "5551234567",
    "message": "Hello! This is a reminder about your appointment tomorrow."
  }'
```

**Response:**

```json
{
  "call_id": "out-1732945860-1",
  "status": "queued",
  "message": "Call initiated"
}
```
![](https://github.com/MaxwellDPS/docs/blob/v1.0/photos/Screenshot%202025-11-29%20214700.png?raw=true)
### 🌅 Morning Weather Briefing

Schedule a daily weather call at 7am:

```bash
curl -X POST http://sip-agent:8080/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "5551234567",
    "tool": "WEATHER",
    "at_time": "07:00",
    "timezone": "America/Los_Angeles",
    "recurring": "daily",
    "prefix": "Good morning! Here is your weather update for today.",
    "suffix": "Have a great day!"
  }' | jq
```


**Response:**

```json
{
  "schedule_id": "a1b2c3d4",
  "status": "scheduled",
  "scheduled_for": "2025-12-01T07:00:00-08:00",
  "recurring": "daily"
}
```

### 🔧 List Available Tools

```bash
curl http://localhost:8080/tools | jq '.[].name'
```

**Output:**

```
"WEATHER"
"SET_TIMER"
"CALLBACK"
"HANGUP"
"STATUS"
"CANCEL"
"DATETIME"
"CALC"
"JOKE"
```

---

## 🧠 Recommended Models

### NVIDIA H100 / A100 (80GB HBM)

Data center GPUs with maximum performance.

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | `meta-llama/Llama-3.1-70B-Instruct` | Best quality, fits in single GPU |
| **LLM** | `Qwen/Qwen2.5-72B-Instruct` | Alternative, excellent reasoning |
| **STT** | `Systran/faster-whisper-large-v3` | Best accuracy |
| **TTS** | `af_heart` | Warm, natural voice |

```env
# H100/A100 80GB Configuration
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
LLM_URL=http://localhost:8000/v1
STT_MODEL=Systran/faster-whisper-large-v3
TTS_VOICE=af_heart
```

---

### NVIDIA DGX Spark (128GB Unified)

Grace Blackwell GB10 with shared CPU/GPU memory.

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | `meta-llama/Llama-3.1-70B-Instruct` | Fits in unified memory |
| **LLM** | `Qwen/Qwen2.5-72B-Instruct` | Alternative option |
| **LLM** | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | Reasoning focused |
| **STT** | `Systran/faster-whisper-large-v3` | Best accuracy |
| **TTS** | `af_heart` | Warm, natural voice |

```env
# DGX Spark Configuration (128GB unified memory)
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
LLM_URL=http://localhost:8000/v1
STT_MODEL=Systran/faster-whisper-large-v3
TTS_VOICE=af_heart
```

---

### NVIDIA RTX 5090 (32GB GDDR7)

Next-gen consumer flagship.

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | `Qwen/Qwen2.5-32B-Instruct` | Best fit for 32GB |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` | Faster, lower quality |
| **LLM** | `mistralai/Mistral-Small-24B-Instruct-2501` | Good balance |
| **STT** | `Systran/faster-whisper-large-v3` | Best accuracy |
| **TTS** | `af_heart` | Warm, natural voice |

```env
# RTX 5090 Configuration (32GB VRAM)
LLM_MODEL=Qwen/Qwen2.5-32B-Instruct
LLM_URL=http://localhost:8000/v1
STT_MODEL=Systran/faster-whisper-large-v3
TTS_VOICE=af_heart
```

---

### NVIDIA RTX 4090 (24GB GDDR6X)

Current consumer flagship.

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | `Qwen/Qwen2.5-14B-Instruct` | Best quality for 24GB |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` | Faster option |
| **LLM** | `mistralai/Mistral-7B-Instruct-v0.3` | Good tool calling |
| **STT** | `Systran/faster-whisper-large-v3` | Best accuracy |
| **TTS** | `af_heart` | Warm, natural voice |

```env
# RTX 4090 Configuration (24GB VRAM)
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
LLM_URL=http://localhost:8000/v1
STT_MODEL=Systran/faster-whisper-large-v3
TTS_VOICE=af_heart
```

---

### NVIDIA RTX 3090 / 4080 (24GB / 16GB)

High-end consumer GPUs.

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` | Best for 16-24GB |
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` | Fast alternative |
| **LLM** | `microsoft/Phi-3-medium-4k-instruct` | 14B, good quality |
| **STT** | `Systran/faster-whisper-medium` | Good balance |
| **TTS** | `af_heart` | Warm, natural voice |

```env
# RTX 3090/4080 Configuration (16-24GB VRAM)
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_URL=http://localhost:8000/v1
STT_MODEL=Systran/faster-whisper-medium
TTS_VOICE=af_heart
```

---

### NVIDIA RTX 3080 / 4070 (10-12GB)

Mid-range GPUs.

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` | Best for 10-12GB |
| **LLM** | `microsoft/Phi-3-mini-4k-instruct` | 3.8B, very fast |
| **LLM** | `meta-llama/Llama-3.2-3B-Instruct` | Lightweight |
| **STT** | `Systran/faster-whisper-small` | Low VRAM |
| **TTS** | `af_heart` | Warm, natural voice |

```env
# RTX 3080/4070 Configuration (10-12GB VRAM)
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_URL=http://localhost:8000/v1
STT_MODEL=Systran/faster-whisper-small
TTS_VOICE=af_heart
```

---

### Low-Latency Stack (Any GPU)

Optimized for fastest response times.

```env
# Minimum latency configuration
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
STT_MODEL=Systran/faster-whisper-tiny.en
TTS_VOICE=af_heart
TTS_SPEED=1.1
```

---

### TTS Voice Options

| Voice | Style | Gender | Accent |
|-------|-------|--------|--------|
| `af_heart` | Warm, friendly | Female | American |
| `af_bella` | Professional | Female | American |
| `af_sarah` | Casual | Female | American |
| `af_nicole` | Expressive | Female | American |
| `am_adam` | Neutral | Male | American |
| `am_michael` | Professional | Male | American |
| `bf_emma` | Warm | Female | British |
| `bm_george` | Professional | Male | British |

---

## 🔧 Built-in Tools

| Tool | Description | Example Phrase |
|------|-------------|----------------|
| 🌤️ `WEATHER` | Current weather conditions | *"What's the weather?"* |
| ⏲️ `SET_TIMER` | Set a countdown timer | *"Set a timer for 5 minutes"* |
| 📞 `CALLBACK` | Schedule a callback | *"Call me back in an hour"* |
| 📴 `HANGUP` | End the call | *"Goodbye"* |
| 📋 `STATUS` | Check pending timers | *"What timers do I have?"* |
| ❌ `CANCEL` | Cancel timers/callbacks | *"Cancel my timer"* |
| 🕐 `DATETIME` | Current date and time | *"What time is it?"* |
| 🧮 `CALC` | Math calculations | *"What's 25 times 4?"* |
| 😄 `JOKE` | Tell a joke | *"Tell me a joke"* |
| 🦜 `SIMON_SAYS` | Repeat back verbatim | *"Simon says hello world"* |
| 📚 `KNOWLEDGE` | Search your local documents (RAG) | *"What's our refund policy?"* |
| 🎲 `DICE` / 🪙 `COIN` | Roll dice, flip coins | *"Roll two twenty-sided dice"* |
| ❓ `TRIVIA` | Play a trivia game | *"Ask me a trivia question"* |
| 📖 `STORY` | Tell an original short story | *"Tell me a bedtime story about a fox"* |
| 🔎 `WEB_SEARCH` | Search the web via self-hosted SearxNG | *"Search for tide times in Monterey"* |
| 🌦️ `FORECAST` | NWS forecast for your coordinates | *"What's the forecast for tomorrow?"* |
| 🌌 `KP_INDEX` | Geomagnetic activity / aurora odds (NOAA) | *"What's the Kp index?"* |
| 🌎 `QUAKES` | Recent earthquakes (USGS) | *"Any earthquakes near me today?"* |
| 🧠 `REMEMBER` / `FORGET` | Manage what the agent remembers about you | *"Remember my gate code is 1234"* |
| ⚡ `TRIGGER_WORKFLOW` | Fire a registered n8n automation | *"Run the lights off automation"* |
| 🖥️ `GPU_STATUS` | GPU utilization/temp from Prometheus | *"How's the GPU doing?"* |
| 🚨 `ALERTS` | Firing monitoring alerts | *"Any alerts firing?"* |
| 🐳 `CONTAINER_CTL` | Status/restart an allowlisted container | *"Restart n8n"* |
| ☎️ `TRANSFER` | Blind-transfer the caller (SIP REFER) | *"Transfer me to extension 2001"* |

Several tools stay hidden until configured: `WEB_SEARCH` needs `SEARXNG_URL`
(started by default in the base compose; `--profile search` on `docker-compose.dgx.yml`), `FORECAST` needs `WEATHER_LATITUDE`/`WEATHER_LONGITUDE`,
`GPU_STATUS`/`ALERTS` need the observability stack, and `CONTAINER_CTL` needs **both** a
`CONTAINER_CTL_ALLOWLIST` and the (commented-out) docker-socket mount — the socket is
root-equivalent on the host, so enable it deliberately.

---

## 🧠 Smarter Conversations

Four opt-in intelligence features (see `.env.example` for every knob):

- **Agentic engine** — set `LLM_BACKEND=langgraph` (plus `LLM_TOOL_CALLING=native`) and the agent runs a LangGraph ReAct loop that can chain several tool calls in one turn ("what's the weather and set a timer for ten minutes"), bounded by `LLM_MAX_TOOL_ROUNDS` / `LLM_AGENT_TIMEOUT_S`. vLLM must be started with tool-call parsing: the compose files pass `VLLM_TOOL_ARGS` (default hermes parser for Qwen3; use `--tool-call-parser openai` for gpt-oss models). Missing deps or runtime errors fall back to the classic engine / a spoken error phrase — never a dead call.
- **Caller memory** — after each call, durable facts about the caller (name, preferences, open items) are distilled into `data/caller_memory/<caller>.json` and recalled the next time they call. On by default (`CALLER_MEMORY_ENABLED`).
- **Knowledge base (RAG)** — drop `.txt`/`.md` files into `data/knowledge/` and restart: they're chunked, embedded locally on CPU (fastembed), and searchable by the `KNOWLEDGE` tool in any tool-calling mode. `KNOWLEDGE_AUTO_INJECT=true` additionally injects top matches into every turn's prompt.
- **Rolling summary** — long calls no longer forget their beginning: turns that fall out of the `MAX_CONVERSATION_TURNS` window are folded into a running summary in the background (`CONVERSATION_SUMMARY_ENABLED`).

---

## 🔌 Creating Plugins

Add custom tools by creating Python plugins:

```python
# src/plugins/hello_tool.py
from tool_plugins import BaseTool, ToolResult, ToolStatus

class HelloTool(BaseTool):
    name = "HELLO"
    description = "Say hello to someone"
    
    parameters = {
        "name": {
            "type": "string",
            "description": "Name to greet",
            "required": True
        }
    }
    
    async def execute(self, params):
        name = params.get("name", "friend")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Hello, {name}! Nice to meet you."
        )
```

Register in `tool_manager.py`:

```python
from plugins.hello_tool import HelloTool

tool_classes = [
    # ... existing tools ...
    HelloTool,
]
```

📖 See [Creating Plugins](https://sip-agent.readme.io/docs/plugins) for the full guide.

---

## 📊 Monitoring

### View Logs

```bash
# Docker logs
docker logs -f sip-agent

# Formatted log viewer
python tools/view-logs.py -f
```

**Example output:**

```
┌──────────────────────────────────────────────────────────────
│ 📞 CALL #1 - From: 1001
└──────────────────────────────────────────────────────────────
15:30:05  📞 Call started
15:30:06  👤 "What's the weather?"
15:30:07  🔧 [TOOL:WEATHER]
15:30:08  🤖 "At Storm Lake, it's 44 degrees..."
15:30:12  👤 "Thanks, goodbye"
15:30:13  📴 Call ended (duration: 0:08)
```

### Grafana Dashboard

Import the included dashboard:

```bash
grafana/dashboards/sip-agent.json
```

![Alt text](https://github.com/MaxwellDPS/docs/blob/v1.0/photos/Screenshot%202025-11-29%20222658.png?raw=true "Dashboard")
![Alt text](https://github.com/MaxwellDPS/docs/blob/v1.0/photos/Screenshot%202025-11-29%20222729.png?raw=true "Dashboard")
![Alt text](https://github.com/MaxwellDPS/docs/blob/v1.0/photos/Screenshot%202025-11-29%20222739.png?raw=true "Dashboard")
![Alt text](https://github.com/MaxwellDPS/docs/blob/v1.0/photos/Screenshot%202025-11-29%20222748.png?raw=true "Dashboard")
![Alt text](https://github.com/MaxwellDPS/docs/blob/v1.0/photos/Screenshot%202025-11-29%20222808.png?raw=true "Dashboard")


---

## 🗂️ Project Structure

```
sip-agent/
├── 📄 README.md                    # 👈 You are here
├── 📄 RELEASE.md                   # Release notes
├── 📄 CHANGELOG.md                 # Version history
├── 📄 docker-compose.yml           # Main compose file
├── 📄 docker-compose.observability.yml
├── 📄 openapi.yaml                 # API specification
│
├── 📂 sip-agent/                   # Core application
│   ├── 📄 Dockerfile
│   ├── 📄 requirements.txt
│   ├── 📄 .env.example
│   ├── 📂 data/
│   │   └── 📄 phrases.json.example
│   └── 📂 src/
│       ├── 📄 main.py              # Application entry
│       ├── 📄 config.py            # Configuration
│       ├── 📄 api.py               # REST API
│       ├── 📄 sip_handler.py       # SIP call handling
│       ├── 📄 audio_pipeline.py    # STT/TTS processing
│       ├── 📄 llm_engine.py        # LLM integration
│       ├── 📄 tool_manager.py      # Tool orchestration
│       ├── 📄 tool_plugins.py      # Plugin base classes
│       ├── 📄 call_queue.py        # Redis call queue
│       ├── 📄 realtime_client.py   # WebSocket STT
│       ├── 📄 telemetry.py         # OpenTelemetry
│       ├── 📄 logging_utils.py     # Structured logging
│       ├── 📄 retry_utils.py       # API retry logic
│       └── 📂 plugins/             # Built-in tools
│           ├── 📄 weather_tool.py
│           ├── 📄 timer_tool.py
│           ├── 📄 callback_tool.py
│           ├── 📄 hangup_tool.py
│           ├── 📄 status_tool.py
│           ├── 📄 cancel_tool.py
│           ├── 📄 datetime_tool.py
│           ├── 📄 calc_tool.py
│           ├── 📄 joke_tool.py
│           └── 📄 simon_says_tool.py
│
├── 📂 docs/                        # Documentation
│   ├── 📄 index.md                 # Overview
│   ├── 📄 getting-started.md       # Installation
│   ├── 📄 configuration.md         # Config reference
│   ├── 📄 api-reference.md         # REST API
│   ├── 📄 tools.md                 # Built-in tools
│   ├── 📄 plugins.md               # Plugin development
│   ├── 📄 examples.md              # Integration examples
│   └── 📂 screenshots/
│
├── 📂 observability/               # Monitoring stack
│   ├── 📂 grafana/
│   │   └── 📂 provisioning/
│   │       ├── 📂 dashboards/      # Pre-built dashboards
│   │       └── 📂 datasources/
│   ├── 📂 prometheus/
│   │   └── 📄 prometheus.yaml
│   ├── 📂 loki/
│   │   └── 📄 loki.yaml
│   ├── 📂 tempo/
│   │   └── 📄 tempo.yaml
│   └── 📂 otel-collector/
│       └── 📄 config.yaml
│
├── 📂 tools/                       # Utilities
│   └── 📄 view-logs.py             # Log viewer
│
└── 📂 .github/
    └── 📂 workflows/
        ├── 📄 docker-build.yml     # Docker CI
        └── 📄 readme-sync.yml      # Docs sync
```

---

## 🖥️ Runs on NVIDIA DGX Spark

This project is optimized to run on the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) with Grace Blackwell architecture.

```
┌─────────────────────────────────────────────────────────────┐
│ 🟢 NVIDIA DGX Spark                                         │
├─────────────────────────────────────────────────────────────┤
│ 🧠 Grace Blackwell GB10 Superchip                          │
│ 💾 128GB Unified Memory                                     │
│ ⚡ 1 PFLOP AI Performance                                   │
├─────────────────────────────────────────────────────────────┤
│ ✅ Local LLM inference (vLLM, Ollama)                      │
│ ✅ Local STT/TTS (Speaches + Whisper + Kokoro)             │
│ ✅ Real-time voice processing                               │
│ ✅ Multiple concurrent calls                                │
└─────────────────────────────────────────────────────────────┘
```

**Recommended DGX Spark setup:**

```env
# Run everything locally on DGX Spark
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=openai/gpt-oss-20b
SPEACHES_API_URL=http://localhost:8001
```

---

## 📖 Documentation

**📚 Full documentation available at [sip-agent.readme.io](https://sip-agent.readme.io)**

| Document | Description |
|----------|-------------|
| [📖 Overview](https://sip-agent.readme.io/docs/overview) | Architecture and features |
| [🚀 Getting Started](https://sip-agent.readme.io/docs/getting-started) | Installation guide |
| [⚙️ Configuration](https://sip-agent.readme.io/docs/configuration) | Environment variables |
| [🌐 API Reference](https://sip-agent.readme.io/docs/api-reference) | REST API endpoints |
| [🔧 Built-in Tools](https://sip-agent.readme.io/docs/tools) | Available tools |
| [🔌 Creating Plugins](https://sip-agent.readme.io/docs/plugins) | Custom tool development |
| [📖 Examples](https://sip-agent.readme.io/docs/examples) | Integration patterns |

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

```bash
# Fork and clone
git clone https://github.com/your-username/sip-agent.git

# Create branch
git checkout -b feature/amazing-feature

# Make changes and test
docker compose up -d
python -m pytest

# Commit with emoji
git commit -m "✨ feat: add amazing feature"

# Push and PR
git push origin feature/amazing-feature
```


---

## 📜 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

```
SPDX-License-Identifier: AGPL-3.0-or-later
```

---

## 🙏 Acknowledgments

- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) — AI supercomputer platform
- [Speaches](https://github.com/speaches-ai/speaches) — Unified STT/TTS server
- [PJSIP](https://www.pjsip.org/) — SIP stack
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [WeatherFlow Tempest](https://tempestwx.com/) — Weather data

---

## 📞 Support

| Resource | Link |
|----------|------|
| 📖 Docs | **[sip-agent.readme.io](https://sip-agent.readme.io)** |
| 🐛 Issues | [GitHub Issues](https://github.com/your-org/sip-agent/issues) |
| 💬 Discussions | [GitHub Discussions](https://github.com/your-org/sip-agent/discussions) |

---

<p align="center">
  Made with ❤️ and 🤖
</p>
