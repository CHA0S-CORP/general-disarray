---
title: "API Reference"
excerpt: "REST API endpoints for the SIP AI Assistant"
category: "API"
slug: "api-reference"
---

# API Reference

The SIP AI Assistant exposes a REST API for initiating calls, executing tools, and managing scheduled tasks.

**Base URL:** `http://your-server:8080`

---

## Health & Status

### GET /health

Check service health and SIP registration status.

**Response:**

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

### GET /queue

Get call queue status.

**Response:**

```json
{
  "enabled": true,
  "pending": 2,
  "active": 1,
  "max_concurrent": 1,
  "total_processed": 47
}
```

---

## Outbound Calls

### POST /call

Initiate an outbound notification call.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | Message to speak |
| `extension` | string | Yes | Phone number or SIP extension |
| `callback_url` | string | No | Webhook for results |
| `ring_timeout` | integer | No | Seconds to wait for answer (default: 30) |
| `call_id` | string | No | Custom call ID for tracking |
| `choice` | object | No | Choice collection config |

**Simple Notification:**

```bash
curl -X POST http://localhost:8080/call \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! This is a reminder about your appointment tomorrow at 2pm.",
    "extension": "5551234567"
  }'
```

**Response:**

```json
{
  "call_id": "out-1732945860-1",
  "status": "queued",
  "message": "Call initiated",
  "queue_position": null
}
```

**With Choice Collection:**

```bash
curl -X POST http://localhost:8080/call \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! This is a reminder about your appointment tomorrow at 2pm.",
    "extension": "5551234567",
    "callback_url": "https://example.com/webhook",
    "choice": {
      "prompt": "Say yes to confirm or no to cancel.",
      "options": [
        {"value": "confirmed", "synonyms": ["yes", "yeah", "yep", "confirm"]},
        {"value": "cancelled", "synonyms": ["no", "nope", "cancel"]}
      ],
      "timeout_seconds": 15
    }
  }'
```

### GET /call/{call_id}

Get status of a specific call.

**Response:**

```json
{
  "call_id": "out-1732945860-1",
  "status": "completed",
  "queued_at": "2025-01-15T10:30:00Z",
  "started_at": "2025-01-15T10:30:05Z",
  "completed_at": "2025-01-15T10:30:45Z",
  "error": null
}
```

---

## Tools

### GET /tools

List all available tools.

**Response:**

```json
[
  {
    "name": "WEATHER",
    "description": "Get current weather conditions from the local weather station",
    "parameters": {},
    "enabled": true
  },
  {
    "name": "SET_TIMER",
    "description": "Set a timer or reminder for a specified duration",
    "parameters": {
      "duration": {"type": "integer", "required": true},
      "message": {"type": "string", "required": false}
    },
    "enabled": true
  }
]
```

### GET /tools/{tool_name}

Get details about a specific tool.

```bash
curl http://localhost:8080/tools/WEATHER
```

### POST /tools/{tool_name}/execute

Execute a tool and get the result.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool` | string | Yes | Tool name |
| `params` | object | No | Tool parameters |
| `speak_result` | boolean | No | Speak result to active call |
| `call_id` | string | No | Specific call to speak to |

**Example - Get Weather:**

```bash
curl -X POST http://localhost:8080/tools/WEATHER/execute \
  -H "Content-Type: application/json" \
  -d '{"tool": "WEATHER"}'
```

**Response:**

```json
{
  "success": true,
  "tool": "WEATHER",
  "message": "At Storm Lake, as of 9:30 pm, it's 44 degrees with foggy conditions. Wind is calm.",
  "data": {
    "temp_f": 44,
    "feels_like_f": 44,
    "humidity": 98,
    "wind_mph": 0,
    "pressure_trend": "steady"
  },
  "spoken": false,
  "error": null
}
```

**Example - Calculate:**

```bash
curl -X POST http://localhost:8080/tools/CALC/execute \
  -H "Content-Type: application/json" \
  -d '{"tool": "CALC", "params": {"expression": "25 * 4 + 10"}}'
```

### POST /tools/{tool_name}/call

Execute a tool and call someone with the result.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool` | string | Yes | Tool name |
| `params` | object | No | Tool parameters |
| `extension` | string | Yes | Phone number to call |
| `prefix` | string | No | Message before tool result |
| `suffix` | string | No | Message after tool result |
| `ring_timeout` | integer | No | Seconds to wait for answer |
| `callback_url` | string | No | Webhook for call results |

**Example - Weather Call:**

```bash
curl -X POST http://localhost:8080/tools/WEATHER/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "WEATHER",
    "extension": "5551234567",
    "prefix": "Good morning! Here is your weather update.",
    "suffix": "Have a great day!"
  }'
```

**Response:**

```json
{
  "call_id": "out-1732945860-2",
  "status": "initiated",
  "tool": "WEATHER",
  "tool_success": true,
  "tool_message": "At Storm Lake, as of 7:00 am, it's 38 degrees...",
  "message": "Calling 5551234567 with WEATHER result"
}
```

---

## Scheduled Calls

### POST /schedule

Schedule a call for a future time.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `extension` | string | Yes | Phone number to call |
| `message` | string | No* | Static message to speak |
| `tool` | string | No* | Tool to execute at call time |
| `tool_params` | object | No | Parameters for the tool |
| `delay_seconds` | integer | No** | Seconds from now |
| `at_time` | string | No** | ISO datetime or HH:MM |
| `timezone` | string | No | Timezone (default: America/Los_Angeles) |
| `prefix` | string | No | Message before tool result |
| `suffix` | string | No | Message after tool result |
| `recurring` | string | No | `daily`, `weekdays`, `weekends` |
| `callback_url` | string | No | Webhook for results |

*Either `message` or `tool` required  
**Either `delay_seconds` or `at_time` required

**Example - Weather in 8 Hours:**

```bash
curl -X POST http://localhost:8080/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "1001",
    "tool": "WEATHER",
    "delay_seconds": 28800,
    "prefix": "Good morning! Here is your weather."
  }'
```

**Example - Daily 7am Weather:**

```bash
curl -X POST http://localhost:8080/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "1001",
    "tool": "WEATHER",
    "at_time": "07:00",
    "timezone": "America/Los_Angeles",
    "recurring": "daily",
    "prefix": "Good morning!"
  }'
```

**Example - One-time Reminder:**

```bash
curl -X POST http://localhost:8080/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "5551234567",
    "message": "This is your reminder to take your medication.",
    "at_time": "2025-01-15T09:00:00",
    "timezone": "America/New_York"
  }'
```

**Response:**

```json
{
  "schedule_id": "a1b2c3d4",
  "status": "scheduled",
  "extension": "1001",
  "scheduled_for": "2025-01-15T07:00:00-08:00",
  "delay_seconds": 28800,
  "message": "Call scheduled for 2025-01-15T07:00:00-08:00",
  "recurring": "daily"
}
```

### GET /schedule

List all scheduled calls.

**Response:**

```json
[
  {
    "schedule_id": "a1b2c3d4",
    "extension": "1001",
    "scheduled_for": "2025-01-15T07:00:00-08:00",
    "remaining_seconds": 25200,
    "message": null,
    "tool": "WEATHER",
    "recurring": "daily",
    "status": "pending"
  }
]
```

### GET /schedule/{schedule_id}

Get details of a scheduled call.

### DELETE /schedule/{schedule_id}

Cancel a scheduled call.

```bash
curl -X DELETE http://localhost:8080/schedule/a1b2c3d4
```

**Response:**

```json
{
  "success": true,
  "message": "Scheduled call a1b2c3d4 cancelled"
}
```

---

## Speak to Active Call

### POST /speak

Inject a message into an active call.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | Text to speak |
| `call_id` | string | No | Specific call ID |

```bash
curl -X POST "http://localhost:8080/speak?message=Attention:%20severe%20weather%20warning"
```

**Response:**

```json
{
  "success": true,
  "message": "Message spoken to call"
}
```

---

## Webhooks

When you provide a `callback_url`, results are POSTed as JSON:

**Outbound Call Webhook:**

```json
{
  "call_id": "out-1732945860-1",
  "status": "completed",
  "extension": "5551234567",
  "duration_seconds": 45.2,
  "message_played": true,
  "choice_response": "confirmed",
  "choice_raw_text": "yes that works",
  "error": null
}
```

**Scheduled Call Webhook:**

```json
{
  "schedule_id": "a1b2c3d4",
  "status": "completed",
  "extension": "1001",
  "tool": "WEATHER",
  "recurring": "daily",
  "timestamp": "2025-01-15T07:00:45Z"
}
```

---

## Error Responses

All errors return appropriate HTTP status codes with details:

```json
{
  "detail": "Tool 'INVALID' not found. Use GET /tools to list available tools."
}
```

| Status | Description |
|--------|-------------|
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 500 | Internal server error |

---

## Rate Limits

The API does not enforce rate limits by default. The call queue limits concurrent outbound calls (configurable via `MAX_CONCURRENT_CALLS`).
