---
title: "Built-in Tools"
excerpt: "Available tools for the voice assistant"
category: "Features"
slug: "tools"
---

# Built-in Tools

The SIP AI Assistant includes several built-in tools that the LLM can invoke during conversations. Tools are triggered when the user asks for something that requires an action.

## How Tools Work

1. User speaks: *"Set a timer for 5 minutes"*
2. LLM recognizes this requires the SET_TIMER tool
3. LLM responds with: `[TOOL:SET_TIMER:duration=300]`
4. Tool executes and returns a result message
5. Assistant speaks the result: *"Timer set for 5 minutes"*

---

## WEATHER

Get current weather conditions from a Tempest weather station.

**Configuration Required:**
```env
TEMPEST_STATION_ID=your-station-id
TEMPEST_API_TOKEN=your-api-token
```

**Trigger Phrases:**
- "What's the weather?"
- "How's the weather outside?"
- "What's the temperature?"
- "Is it raining?"

**Example Output:**
> "At Storm Lake, as of 9:30 pm, it's 44 degrees with foggy conditions. Wind is calm. Yesterday saw half an inch of rain."

**Features:**
- Temperature with feels-like
- Humidity and fog detection
- Wind speed, direction, and gusts
- Precipitation (current, today, yesterday)
- Lightning detection with distance
- UV index warnings
- Barometric pressure trends

---

## SET_TIMER

Set a timer that fires during or after the call.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `duration` | integer | Yes | - | Duration in seconds |
| `message` | string | No | "Your timer is complete" | Message when timer fires |

**Trigger Phrases:**
- "Set a timer for 5 minutes"
- "Remind me in 30 seconds"
- "Set a 2 hour timer"

**Example:**
> User: "Set a timer for 10 minutes for my pizza"  
> Assistant: "Timer set for 10 minutes"  
> *(10 minutes later)*  
> Assistant: "Your pizza timer is complete!"

**Limits:**
- Maximum duration: 24 hours (configurable via `MAX_TIMER_DURATION_HOURS`)

---

## CALLBACK

Schedule a callback - the assistant will call you back later.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `delay` | integer | No | 60 | Delay in seconds |
| `message` | string | No | "This is your callback" | Message to speak |
| `destination` | string | No | Current caller | Phone number to call |

**Trigger Phrases:**
- "Call me back in 30 minutes"
- "Remind me to check the oven in 20 minutes"
- "Schedule a callback for 5pm"

**Example:**
> User: "Call me back in an hour to remind me about my meeting"  
> Assistant: "I'll call you back in 1 hour"  
> *(1 hour later, phone rings)*  
> Assistant: "This is your reminder about your meeting"

---

## HANGUP

End the current call gracefully.

**No Parameters**

**Trigger Phrases:**
- "Goodbye"
- "Hang up"
- "End the call"
- "That's all, thanks"

The assistant will say goodbye before disconnecting.

---

## STATUS

Check status of pending timers and scheduled callbacks.

**No Parameters**

**Trigger Phrases:**
- "What timers do I have?"
- "Check my callbacks"
- "What's scheduled?"

**Example:**
> User: "What timers do I have running?"  
> Assistant: "You have 1 timer: 4 minutes remaining for your pizza. You also have a callback scheduled in 45 minutes."

---

## CANCEL

Cancel pending timers or callbacks.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task_type` | string | No | "all" | "timer", "callback", or "all" |

**Trigger Phrases:**
- "Cancel my timer"
- "Cancel all timers"
- "Cancel my callback"

---

## DATETIME

Get the current date and/or time.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `format` | string | No | "datetime" | "time", "date", "datetime", "full" |
| `timezone` | string | No | System TZ | Timezone name |

**Trigger Phrases:**
- "What time is it?"
- "What's today's date?"
- "What day is it?"

**Format Examples:**
- `time`: "It's 3:45 PM"
- `date`: "Today is Saturday, January 15th, 2025"
- `datetime`: "It's Saturday, January 15th at 3:45 PM"
- `full`: "It's Saturday, January 15th, 2025 at 3:45:30 PM Pacific Standard Time"

---

## CALC

Perform mathematical calculations.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `expression` | string | Yes | Math expression to evaluate |

**Supported Operations:**
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Integer division: `//`
- Modulo: `%`
- Exponentiation: `**`
- Parentheses: `()`

**Trigger Phrases:**
- "What's 25 times 4?"
- "Calculate 15% of 200"
- "What's 144 divided by 12?"

**Example:**
> User: "What's 25 percent of 80?"  
> Assistant: "25 percent of 80 is 20"

**Safety:**
- Uses AST parsing (not `eval()`)
- Only allows numeric operations
- No code execution possible

---

## JOKE

Tell a random joke.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `category` | string | No | "general" | "general", "tech", "dad" |

**Trigger Phrases:**
- "Tell me a joke"
- "Got any dad jokes?"
- "Tell me a tech joke"

---

## Tool Invocation Format

The LLM invokes tools using this format in its response:

```
[TOOL:TOOL_NAME]
[TOOL:TOOL_NAME:param1=value1,param2=value2]
```

**Examples:**
```
[TOOL:WEATHER]
[TOOL:SET_TIMER:duration=300,message=Pizza is ready]
[TOOL:CALLBACK:delay=3600,message=Meeting reminder]
[TOOL:CALC:expression=25*4]
[TOOL:DATETIME:format=full,timezone=America/New_York]
```

---

## Enabling/Disabling Tools

Tools can be enabled or disabled via environment variables:

```env
ENABLE_TIMER_TOOL=true
ENABLE_CALLBACK_TOOL=true
ENABLE_WEATHER_TOOL=true
```

When disabled, the tool won't appear in the LLM's system prompt and won't be available.

---

## Creating Custom Tools

See [Creating Plugins](plugins) to add your own tools.
