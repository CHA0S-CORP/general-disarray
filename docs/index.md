---
title: "SIP AI Assistant"
excerpt: "Voice-powered AI assistant for SIP phone systems"
category: "Overview"
slug: "overview"
---

# SIP AI Assistant

A voice-powered AI assistant that answers phone calls, understands natural language, and can perform actions like setting timers, checking weather, scheduling callbacks, and more.

## Features

- **Voice Conversations** - Natural speech-to-text and text-to-speech powered by Whisper and XTTS/Piper
- **LLM Integration** - Connects to any OpenAI-compatible API (vLLM, LM Studio, OpenAI, etc.)
- **Built-in Tools** - Weather, timers, callbacks, date/time, calculator, and more
- **Plugin System** - Easily add custom tools by dropping Python files in a directory
- **REST API** - Initiate outbound calls, execute tools, schedule calls via HTTP
- **Webhooks** - Trigger calls from external systems with tool results
- **Scheduled Calls** - Set up one-time or recurring calls (daily weather briefings, reminders)
- **Call Queue** - Manage concurrent calls with configurable limits
- **Observability** - Prometheus metrics, OpenTelemetry tracing, structured JSON logs

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   SIP Phone     │────▶│  SIP AI Agent   │────▶│   LLM Server    │
│   (Caller)      │◀────│                 │◀────│  (vLLM/OpenAI)  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Whisper  │ │   TTS    │ │  Tools   │
              │  (STT)   │ │(XTTS/etc)│ │ Plugins  │
              └──────────┘ └──────────┘ └──────────┘
```

## Quick Example

Call the assistant and say:

> "What's the weather like?"

The assistant will:
1. Transcribe your speech using Whisper
2. Send the text to the LLM
3. LLM decides to call the WEATHER tool
4. Tool fetches data from your Tempest weather station
5. LLM speaks the weather report back to you

> "At Storm Lake, as of 9:30 pm, it's 44 degrees with foggy conditions. Wind is calm. Yesterday saw half an inch of rain."

## Use Cases

- **Smart Home Integration** - "Set a timer for 10 minutes" / "Call me back in an hour"
- **Weather Briefings** - Scheduled morning calls with weather updates
- **Appointment Reminders** - Outbound calls with confirmation prompts
- **Notifications** - Webhook-triggered calls for alerts and announcements
- **Voice Control** - Hands-free control of home automation systems

## Getting Started

1. [Installation & Setup](getting-started)
2. [Configuration Reference](configuration)
3. [API Documentation](api-reference)
4. [Available Tools](tools)
5. [Creating Plugins](plugins)
