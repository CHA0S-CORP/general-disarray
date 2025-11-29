#!/bin/bash
# tail-logs.sh - Tail SIP AI Assistant logs with filtered interesting events
#
# Usage: ./tail-logs.sh [container_name]
#        docker compose logs -f sip-ai-assistant | ./tail-logs.sh
#
# Interesting events shown:
#   - call_start, call_end, call_timeout
#   - user_speech, assistant_response, assistant_ack
#   - timer_set, timer_fired, callback_scheduled, callback_execute, callback_complete
#   - tool_call, barge_in
#   - Errors and warnings

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Events to show
INTERESTING_EVENTS=(
    "call_start"
    "call_end"
    "call_timeout"
    "user_speech"
    "assistant_response"
    "assistant_ack"
    "timer_set"
    "timer_fired"
    "callback_scheduled"
    "callback_execute"
    "callback_complete"
    "tool_call"
    "barge_in"
    "task_execute"
)

# Build grep pattern
EVENT_PATTERN=$(IFS="|"; echo "${INTERESTING_EVENTS[*]}")

format_log() {
    local line="$1"
    
    # Try to parse JSON
    if echo "$line" | grep -q '"event"'; then
        # Extract fields using grep/sed (portable)
        local ts=$(echo "$line" | grep -o '"ts":"[^"]*"' | cut -d'"' -f4)
        local level=$(echo "$line" | grep -o '"level":"[^"]*"' | cut -d'"' -f4)
        local event=$(echo "$line" | grep -o '"event":"[^"]*"' | cut -d'"' -f4)
        local msg=$(echo "$line" | grep -o '"msg":"[^"]*"' | cut -d'"' -f4)
        local data=$(echo "$line" | grep -o '"data":{[^}]*}' | sed 's/"data"://')
        
        # Color based on event type
        local color="$WHITE"
        local icon=""
        
        case "$event" in
            call_start)
                color="$GREEN"
                icon="📞"
                ;;
            call_end|call_timeout)
                color="$RED"
                icon="📴"
                ;;
            user_speech)
                color="$CYAN"
                icon="🎤"
                ;;
            assistant_response|assistant_ack)
                color="$MAGENTA"
                icon="🤖"
                ;;
            timer_set|timer_fired)
                color="$YELLOW"
                icon="⏰"
                ;;
            callback_scheduled|callback_execute|callback_complete)
                color="$BLUE"
                icon="📲"
                ;;
            barge_in)
                color="$YELLOW"
                icon="✋"
                ;;
            tool_call|task_execute)
                color="$WHITE"
                icon="🔧"
                ;;
            *)
                icon="•"
                ;;
        esac
        
        # Format timestamp (extract time portion)
        local time_only=$(echo "$ts" | grep -o '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]' || echo "$ts")
        
        # Print formatted output
        printf "${GRAY}%s${NC} ${color}%s %-20s${NC} %s" "$time_only" "$icon" "[$event]" "$msg"
        
        # Print data if present
        if [ -n "$data" ] && [ "$data" != "{}" ]; then
            printf " ${GRAY}%s${NC}" "$data"
        fi
        printf "\n"
        
    elif echo "$line" | grep -qiE '(error|warning|warn)'; then
        # Show errors/warnings with color
        if echo "$line" | grep -qi 'error'; then
            printf "${RED}%s${NC}\n" "$line"
        else
            printf "${YELLOW}%s${NC}\n" "$line"
        fi
    fi
}

# Header
echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${WHITE}  SIP AI Assistant - Filtered Event Log${NC}"
echo -e "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GRAY}Events: ${EVENT_PATTERN}${NC}"
echo -e "${GRAY}Plus: errors and warnings${NC}"
echo -e "${WHITE}───────────────────────────────────────────────────────────────${NC}"
echo ""

# Process input
if [ $# -ge 1 ]; then
    # Container name provided - run docker logs
    docker logs -f "$1" 2>&1 | while IFS= read -r line; do
        # Filter for interesting events or errors
        if echo "$line" | grep -qE "(\"event\":\"($EVENT_PATTERN)\"|error|warning|warn)" 2>/dev/null; then
            format_log "$line"
        fi
    done
else
    # Read from stdin (piped input)
    while IFS= read -r line; do
        # Filter for interesting events or errors
        if echo "$line" | grep -qE "(\"event\":\"($EVENT_PATTERN)\"|error|warning|warn)" 2>/dev/null; then
            format_log "$line"
        fi
    done
fi
