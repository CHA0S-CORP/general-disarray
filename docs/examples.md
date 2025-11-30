---
title: "Examples"
excerpt: "Common use cases and integration examples"
category: "Guides"
slug: "examples"
---

# Examples

Real-world examples and integration patterns for the SIP AI Assistant.

---

## Morning Weather Briefing

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
  }'
```

---

## Appointment Reminders

Send appointment reminders with confirmation:

```bash
curl -X POST http://sip-agent:8080/call \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "5551234567",
    "message": "Hello! This is a reminder that you have an appointment with Dr. Smith tomorrow at 2pm.",
    "callback_url": "https://your-app.com/webhook/appointment",
    "choice": {
      "prompt": "Would you like to confirm or cancel this appointment?",
      "options": [
        {"value": "confirmed", "synonyms": ["yes", "confirm", "keep", "sounds good"]},
        {"value": "cancelled", "synonyms": ["no", "cancel", "reschedule"]}
      ],
      "timeout_seconds": 20,
      "repeat_count": 2
    }
  }'
```

Handle the webhook:

```python
@app.post("/webhook/appointment")
async def handle_appointment_webhook(payload: dict):
    call_id = payload["call_id"]
    choice = payload.get("choice_response")
    
    if choice == "confirmed":
        # Mark appointment confirmed in your system
        await confirm_appointment(call_id)
    elif choice == "cancelled":
        # Open slot for rebooking
        await cancel_appointment(call_id)
    else:
        # No response or unclear - follow up later
        await schedule_followup(call_id)
```

---

## Home Assistant Integration

Trigger a weather call when leaving home:

**Home Assistant automation.yaml:**

```yaml
automation:
  - alias: "Weather announcement when leaving"
    trigger:
      - platform: state
        entity_id: person.john
        from: "home"
        to: "not_home"
    condition:
      - condition: time
        after: "06:00:00"
        before: "10:00:00"
    action:
      - service: rest_command.weather_call
```

**Home Assistant configuration.yaml:**

```yaml
rest_command:
  weather_call:
    url: "http://sip-agent:8080/tools/WEATHER/call"
    method: POST
    content_type: "application/json"
    payload: >
      {
        "tool": "WEATHER",
        "extension": "5551234567",
        "prefix": "Good morning! Before you head out, here's the weather."
      }
```

---

## Monitoring Alerts

Alert on-call engineer when system goes down:

```bash
# In your monitoring script or alertmanager webhook
curl -X POST http://sip-agent:8080/call \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "5551234567",
    "message": "Alert! The production database server is not responding. CPU usage was at 98% before it went offline. Please investigate immediately.",
    "callback_url": "https://monitoring.example.com/alert-acknowledged",
    "choice": {
      "prompt": "Say acknowledge to confirm you received this alert.",
      "options": [
        {"value": "acknowledged", "synonyms": ["acknowledge", "ack", "got it", "on it"]}
      ],
      "timeout_seconds": 30
    }
  }'
```

**Alertmanager webhook receiver:**

```yaml
receivers:
  - name: 'phone-alert'
    webhook_configs:
      - url: 'http://alert-bridge:8000/alert'
```

**Alert bridge service:**

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.post("/alert")
async def handle_alert(alert: dict):
    message = f"Alert: {alert['labels']['alertname']}. "
    message += f"{alert['annotations'].get('description', '')}"
    
    async with httpx.AsyncClient() as client:
        await client.post(
            "http://sip-agent:8080/call",
            json={
                "extension": get_oncall_number(),
                "message": message,
                "callback_url": "http://alert-bridge:8000/ack"
            }
        )
```

---

## Medication Reminders

Recurring medication reminders:

```bash
# Morning medication
curl -X POST http://sip-agent:8080/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "5551234567",
    "message": "Good morning! This is your reminder to take your morning medication.",
    "at_time": "08:00",
    "recurring": "daily"
  }'

# Evening medication
curl -X POST http://sip-agent:8080/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "extension": "5551234567",
    "message": "Good evening! This is your reminder to take your evening medication.",
    "at_time": "20:00",
    "recurring": "daily"
  }'
```

---

## Cron-based Weather Calls

Use cron for flexible scheduling:

```bash
# /etc/cron.d/weather-calls

# Weekday mornings at 6:30am
30 6 * * 1-5 root curl -X POST http://sip-agent:8080/tools/WEATHER/call -H "Content-Type: application/json" -d '{"tool":"WEATHER","extension":"1001","prefix":"Good morning! Time to wake up."}' > /dev/null 2>&1

# Weekend mornings at 8am
0 8 * * 0,6 root curl -X POST http://sip-agent:8080/tools/WEATHER/call -H "Content-Type: application/json" -d '{"tool":"WEATHER","extension":"1001","prefix":"Good morning!"}' > /dev/null 2>&1
```

---

## n8n Workflow Integration

Create a workflow that calls when a form is submitted:

```json
{
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "contact-form",
        "httpMethod": "POST"
      }
    },
    {
      "name": "HTTP Request",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://sip-agent:8080/call",
        "jsonParameters": true,
        "bodyParameters": {
          "message": "You have a new contact form submission from {{ $json.name }}. They said: {{ $json.message }}",
          "extension": "1001"
        }
      }
    }
  ]
}
```

---

## Python Integration

```python
import httpx
from typing import Optional

class SIPAssistant:
    def __init__(self, base_url: str = "http://sip-agent:8080"):
        self.base_url = base_url
    
    async def call(
        self,
        extension: str,
        message: str,
        callback_url: Optional[str] = None
    ) -> dict:
        """Make an outbound call."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/call",
                json={
                    "extension": extension,
                    "message": message,
                    "callback_url": callback_url
                }
            )
            return response.json()
    
    async def weather_call(
        self,
        extension: str,
        prefix: Optional[str] = None
    ) -> dict:
        """Call with weather update."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tools/WEATHER/call",
                json={
                    "tool": "WEATHER",
                    "extension": extension,
                    "prefix": prefix
                }
            )
            return response.json()
    
    async def schedule_daily_weather(
        self,
        extension: str,
        time: str,
        timezone: str = "America/Los_Angeles"
    ) -> dict:
        """Schedule daily weather calls."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/schedule",
                json={
                    "extension": extension,
                    "tool": "WEATHER",
                    "at_time": time,
                    "timezone": timezone,
                    "recurring": "daily",
                    "prefix": "Good morning! Here's your weather."
                }
            )
            return response.json()
    
    async def get_weather(self) -> dict:
        """Get weather data without calling."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tools/WEATHER/execute",
                json={"tool": "WEATHER"}
            )
            return response.json()

# Usage
async def main():
    assistant = SIPAssistant()
    
    # Get weather data
    weather = await assistant.get_weather()
    print(f"Weather: {weather['message']}")
    
    # Schedule daily briefing
    result = await assistant.schedule_daily_weather(
        extension="5551234567",
        time="07:00"
    )
    print(f"Scheduled: {result['schedule_id']}")
```

---

## Node.js Integration

```javascript
const axios = require('axios');

class SIPAssistant {
  constructor(baseUrl = 'http://sip-agent:8080') {
    this.baseUrl = baseUrl;
  }

  async call(extension, message, callbackUrl = null) {
    const response = await axios.post(`${this.baseUrl}/call`, {
      extension,
      message,
      callback_url: callbackUrl
    });
    return response.data;
  }

  async weatherCall(extension, prefix = null) {
    const response = await axios.post(`${this.baseUrl}/tools/WEATHER/call`, {
      tool: 'WEATHER',
      extension,
      prefix
    });
    return response.data;
  }

  async scheduleDailyWeather(extension, time, timezone = 'America/Los_Angeles') {
    const response = await axios.post(`${this.baseUrl}/schedule`, {
      extension,
      tool: 'WEATHER',
      at_time: time,
      timezone,
      recurring: 'daily',
      prefix: 'Good morning! Here is your weather.'
    });
    return response.data;
  }
}

// Usage
const assistant = new SIPAssistant();

// Schedule morning weather
assistant.scheduleDailyWeather('5551234567', '07:00')
  .then(result => console.log('Scheduled:', result.schedule_id));
```

---

## Grafana Alert Integration

Configure Grafana to call via webhook:

**Contact Point:**

```yaml
apiVersion: 1
contactPoints:
  - name: phone-alert
    receivers:
      - uid: phone
        type: webhook
        settings:
          url: http://sip-agent:8080/call
          httpMethod: POST
        secureSettings: {}
```

**Notification Template:**

```go
{{ define "phone.message" }}
Alert: {{ .CommonLabels.alertname }}
Status: {{ .Status }}
{{ range .Alerts }}
{{ .Annotations.summary }}
{{ end }}
{{ end }}
```

---

## Docker Health Check

Monitor the assistant and restart if unhealthy:

```yaml
services:
  sip-agent:
    image: sip-agent:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
```
