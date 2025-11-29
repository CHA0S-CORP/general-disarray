# SIP AI Assistant

A voice-based AI assistant that answers phone calls via SIP, powered by local LLM inference. Supports natural conversations, timers, callbacks, and extensible tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Docker Compose                               │
├─────────────────┬─────────────────────┬─────────────────────────────┤
│                 │                     │                             │
│    vLLM         │     Speaches        │      SIP Agent              │
│    Server       │     (STT + TTS)     │      (Orchestrator)         │
│                 │                     │                             │
│  ┌───────────┐  │  ┌───────────────┐  │  ┌───────────────────────┐  │
│  │ LLM Model │  │  │ Whisper (STT) │  │  │  PJSIP Call Handler   │  │
│  │ (Qwen2.5) │  │  │ Kokoro (TTS)  │  │  │  Audio Pipeline       │  │
│  └───────────┘  │  └───────────────┘  │  │  Tool Manager         │  │
│                 │                     │  │  LLM Client            │  │
│  Port: 8000     │  Port: 8001         │  └───────────────────────┘  │
│                 │                     │                             │
└─────────────────┴─────────────────────┴─────────────────────────────┘
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │   SIP Server      │
                                    │   (Asterisk/      │
                                    │    FreeSWITCH)    │
                                    └───────────────────┘
```

**Components:**

| Service | Purpose | Model/Tech |
|---------|---------|------------|
| **vLLM** | LLM inference | Qwen2.5-7B-Instruct (configurable) |
| **Speaches** | Speech-to-Text | Whisper (configurable size) |
| **Speaches** | Text-to-Speech | Kokoro-82M with af_heart voice |
| **SIP Agent** | Orchestration | PJSIP + Python asyncio |

## Features

- **Natural Conversations** - Context-aware multi-turn dialogue
- **Voice Activity Detection** - Automatic speech endpoint detection
- **Barge-in Support** - Interrupt the assistant mid-response
- **Timers** - "Set a timer for 5 minutes"
- **Callbacks** - "Call me back in 10 minutes"
- **Outbound Call API** - REST API for notification calls with response collection
- **Extensible Tools** - Easy to add new capabilities
- **JSON Structured Logging** - Filterable event stream
- **Pre-cached Phrases** - Low-latency greetings and acknowledgments

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- SIP server (Asterisk, FreeSWITCH, etc.)
- ~16GB+ VRAM recommended for default models

## Quick Start

1. **Clone and configure:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Start services:**
   ```bash
   docker compose up -d
   ```

3. **View logs:**
   ```bash
   ./view-logs.py
   ```

4. **Call the SIP extension** configured in your `.env`

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# SIP Configuration
SIP_USER=1000                    # SIP extension/username
SIP_PASSWORD=secret              # SIP password
SIP_DOMAIN=pbx.example.com       # SIP server domain
SIP_REGISTRAR=pbx.example.com    # SIP registrar (optional)

# LLM Configuration
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  # HuggingFace model ID
LLM_API_URL=http://vllm:8000/v1     # vLLM endpoint

# Speech Configuration
SPEACHES_URL=http://speaches:8001   # Speaches API endpoint
WHISPER_MODEL=base                  # tiny, base, small, medium, large
WHISPER_COMPUTE_TYPE=auto           # auto, int8, float16, float32
TTS_MODEL=kokoro                    # TTS model
TTS_VOICE=af_heart                  # Voice preset

# Audio Settings
SAMPLE_RATE=16000                   # Audio sample rate

# Callback Settings
CALLBACK_RING_TIMEOUT=30            # Seconds to wait for answer
CALLBACK_RETRY_ATTEMPTS=2           # Retry failed callbacks
CALLBACK_RETRY_DELAY=30             # Seconds between retries

# Tempest Weather API (optional)
TEMPEST_STATION_ID=12345            # Your Tempest station ID
TEMPEST_API_TOKEN=your-token        # API token from tempestwx.com
```

### System Prompt

Edit `config.py` to customize the assistant's personality and behavior:

```python
SYSTEM_PROMPT = """You are a helpful AI voice assistant...

Available tools:
- TIMER: [TOOL:TIMER:duration=SECONDS,message=TEXT]
- CALLBACK: [TOOL:CALLBACK:delay=SECONDS,message=TEXT]
- WEATHER: [TOOL:WEATHER]
- HANGUP: [TOOL:HANGUP]
"""
```

## Tools

The assistant supports inline tool calls in responses:

### Timer
```
User: "Set a timer for 5 minutes"
Assistant: "I've set a timer for 5 minutes. [TOOL:TIMER:duration=300,message=Your 5 minute timer is done!]"
```

### Callback
```
User: "Call me back in 10 minutes to remind me about the meeting"
Assistant: "I'll call you back in 10 minutes. [TOOL:CALLBACK:delay=600,message=This is your reminder about the meeting]"
```

### Weather
```
User: "What's the weather like?"
Assistant: "Let me check. [TOOL:WEATHER] It's 72 degrees, 45% humidity, with wind from the northwest at 8 mph."
```
Requires `TEMPEST_STATION_ID` and `TEMPEST_API_TOKEN` to be configured. Get these from [tempestwx.com/settings/tokens](https://tempestwx.com/settings/tokens).

### Hangup
```
User: "Goodbye"
Assistant: "Goodbye! Have a great day. [TOOL:HANGUP]"
```

## Outbound Call API

The assistant includes a REST API for initiating outbound notification calls with optional response collection.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/call` | Initiate outbound call |
| GET | `/call/{call_id}` | Check call status |

### Basic Notification Call

Send a message to an extension (no webhook needed for simple notifications):

```bash
curl -X POST http://localhost:8080/call \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, this is a reminder about your appointment tomorrow at 2pm.",
    "extension": "1001",
    "ring_timeout": 30
  }'
```

**Response:**
```json
{
  "call_id": "out-1234567890-1",
  "status": "queued",
  "message": "Call initiated"
}
```

### Call with Choice Collection

Prompt the user for a response (webhook required to receive the choice):

```bash
curl -X POST http://localhost:8080/call \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, this is a reminder about your appointment tomorrow at 2pm.",
    "extension": "1001",
    "callback_url": "https://example.com/webhook",
    "ring_timeout": 30,
    "choice": {
      "prompt": "Would you like to confirm or cancel? Say yes to confirm or no to cancel.",
      "options": [
        {"value": "confirmed", "synonyms": ["yes", "yeah", "yep", "confirm", "okay", "sure"]},
        {"value": "cancelled", "synonyms": ["no", "nope", "cancel", "nevermind"]}
      ],
      "timeout_seconds": 15,
      "repeat_count": 2
    }
  }'
```

### Webhook Payload

After the call completes, a POST request is sent to your `callback_url`:

```json
{
  "call_id": "out-1234567890-1",
  "status": "completed",
  "extension": "1001",
  "duration_seconds": 45.2,
  "message_played": true,
  "choice_response": "confirmed",
  "choice_raw_text": "yes I confirm",
  "error": null
}
```

**Status Values:**
- `completed` - Call answered, message played
- `no_answer` - Call not answered within ring timeout
- `failed` - Call failed to connect
- `busy` - Extension was busy

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | Message to speak to recipient |
| `extension` | string | Yes | SIP extension or full URI |
| `callback_url` | string | If choice | Webhook URL for results (required when using choice) |
| `ring_timeout` | int | No | Seconds to wait for answer (default: 30) |
| `call_id` | string | No | Custom call ID for tracking |
| `choice` | object | No | Choice prompt configuration |

### Choice Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Question to ask the user |
| `options` | array | Yes | List of valid choices |
| `timeout_seconds` | int | No | Wait time for response (default: 30) |
| `repeat_count` | int | No | Times to repeat if no response (default: 2) |

### Choice Option

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `value` | string | Yes | Value returned in webhook |
| `synonyms` | array | No | Alternative phrases that map to this choice |

## Log Viewer

The included `view-logs.py` script provides filtered, formatted log output:

```bash
# View interesting events only (default)
./view-logs.py

# View all logs
./view-logs.py -a

# Tail specific service
./view-logs.py sip-agent
```

**Event Types:**

| Icon | Event | Description |
|------|-------|-------------|
| 🔥 | `warming_up` | Service starting |
| ✅ | `ready` | Service ready |
| 🌐 | `api_started` | API server started |
| 📞 | `call_start` | Incoming/outgoing call |
| 📴 | `call_end` | Call ended |
| 📤 | `outbound_call_initiated` | Outbound call queued |
| 📞 | `outbound_call_answered` | Outbound call answered |
| 🔊 | `outbound_call_message_played` | Message played |
| ✅ | `outbound_call_choice_collected` | User choice collected |
| 🔗 | `outbound_call_webhook` | Webhook sent |
| 🎤 | `user_speech` | User transcription |
| 🤖 | `assistant_response` | LLM response |
| 💬 | `assistant_ack` | Acknowledgment ("Okay", "Got it") |
| 🔧 | `tool_call` | Tool invoked |
| 📋 | `task_scheduled` | Timer/callback scheduled |
| ⏰ | `timer_set` | Timer created |
| 🔔 | `timer_fired` | Timer completed |
| 📲 | `callback_scheduled` | Callback scheduled |
| 🌤️ | `weather_fetch` | Weather data retrieved |
| ⚡ | `task_execute` | Task executing |
| ✋ | `barge_in` | User interrupted |

**Sample Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⟵ CALL #1: sip:420@10.42.252.54
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

00:15:23     🎤 User: Set a timer for 30 seconds
00:15:24     💬 Assistant: Okay
00:15:25       🔧 Tool called: TIMER (params=delay=30, message=...)
00:15:25       📋 Task scheduled: timer in 30s
00:15:26     🤖 Assistant: I've set a timer for 30 seconds
00:15:55       🔔 Timer fired: Your timer is done

──────────────────────────────────────────────────────────────────────
  ✗ CALL ENDED
──────────────────────────────────────────────────────────────────────
```

## Project Structure

```
sip-agent-speaches/
├── docker-compose.yml    # Service definitions
├── Dockerfile            # SIP agent container
├── .env.example          # Configuration template
├── requirements.txt      # Python dependencies
├── main.py               # Main orchestrator
├── api.py                # REST API for outbound calls
├── sip_handler.py        # PJSIP call handling
├── audio_pipeline.py     # STT/TTS via Speaches API
├── llm_engine.py         # LLM client (vLLM/OpenAI API)
├── tool_manager.py       # Timer, callback, tool framework
├── config.py             # Configuration and system prompt
├── view-logs.py          # Log viewer script
└── README.md             # This file
```

## Troubleshooting

### "Requested float16 compute type, but device does not support"
Set `WHISPER_COMPUTE_TYPE=auto` in `.env` to auto-detect the best compute type.

### Call connects but no audio
- Check SIP server NAT settings
- Verify RTP ports are open (10000-10100)
- Check `SIP_DOMAIN` matches your server

### Slow response times
- Use a smaller Whisper model (`tiny` or `base`)
- Ensure GPU is being utilized (check `nvidia-smi`)
- Pre-cache common phrases (enabled by default)

### PJSIP assertion errors on shutdown
These are cosmetic and don't affect operation. The cleanup sequence handles them gracefully.

### Tool calls not working
Ensure the LLM response contains the exact format:
```
[TOOL:TOOLNAME:param1=value1,param2=value2]
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 16GB+ |
| System RAM | 16GB | 32GB |
| CPU | 4 cores | 8+ cores |

**Tested Configurations:**
- NVIDIA RTX 4090 (24GB) - All models
- NVIDIA DGX Spark GB10 - With ARM64 compatibility fixes
- NVIDIA RTX 3080 (10GB) - Smaller models

## Extending

### Adding New Tools

1. Create a tool class in `tool_manager.py`:
   ```python
   class WeatherTool(BaseTool):
       name = "WEATHER"
       description = "Get current weather"
       
       async def execute(self, params: Dict[str, Any]) -> ToolResult:
           location = params.get('location', 'default')
           # Implement weather lookup
           return ToolResult(
               status=ToolStatus.SUCCESS,
               message=f"The weather in {location} is sunny"
           )
   ```

2. Register in `ToolManager.start()`:
   ```python
   self.register_tool(WeatherTool(self.assistant))
   ```

3. Update system prompt in `config.py`:
   ```
   - WEATHER: [TOOL:WEATHER:location=CITY]
   ```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [PJSIP](https://www.pjsip.org/) - SIP stack
- [Speaches](https://github.com/speaches-ai/speaches) - STT/TTS API
- [vLLM](https://github.com/vllm-project/vllm) - LLM inference
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) - Text-to-speech