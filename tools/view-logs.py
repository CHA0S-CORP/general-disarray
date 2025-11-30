#!/usr/bin/env python3
"""
SIP AI Assistant - Log Viewer
Filters and displays interesting events from JSON logs, grouped by call.

Usage:
    ./view-logs.py                          # Tail docker compose logs
    ./view-logs.py sip-agent                # Tail specific service
    ./view-logs.py -a                       # Show ALL logs (not just events)
    docker compose logs -f | ./view-logs.py --stdin
"""

import sys
import json
import argparse
import subprocess

# ANSI colors
class C:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    GRAY = '\033[0;90m'
    BOLD = '\033[1m'
    NC = '\033[0m'

# Event styling: event_name -> (icon, color, category)
# Categories: 'system', 'call', 'speech', 'tool', 'task', 'api'
EVENT_STYLE = {
    # System events
    'warming_up': ('🔥', C.YELLOW, 'system'),
    'ready': ('✅', C.GREEN, 'system'),
    'api_started': ('🌐', C.GREEN, 'system'),
    'queue_started': ('📋', C.GREEN, 'system'),
    'queue_connected': ('📋', C.GREEN, 'system'),
    
    # Call lifecycle
    'call_start': ('📞', C.GREEN, 'call'),
    'call_end': ('📴', C.RED, 'call'),
    'call_timeout': ('📴', C.RED, 'call'),
    
    # Outbound calls
    'outbound_call_initiated': ('📤', C.BLUE, 'outbound'),
    'outbound_call_dialing': ('📞', C.YELLOW, 'outbound'),
    'outbound_call_answered': ('📞', C.GREEN, 'outbound'),
    'outbound_call_no_answer': ('📴', C.RED, 'outbound'),
    'outbound_call_message_played': ('🔊', C.GREEN, 'outbound'),
    'outbound_call_ack_played': ('✅', C.GREEN, 'outbound'),
    'outbound_call_choice_collected': ('✅', C.GREEN, 'outbound'),
    'outbound_call_choice_no_match': ('❓', C.YELLOW, 'outbound'),
    'outbound_call_webhook': ('🔗', C.BLUE, 'outbound'),
    'outbound_call_webhook_success': ('✅', C.GREEN, 'outbound'),
    'outbound_call_complete': ('✅', C.GREEN, 'outbound'),
    'outbound_call_failed': ('❌', C.RED, 'outbound'),
    
    # Call queue
    'call_queued': ('📥', C.BLUE, 'call'),
    'call_waited': ('⏳', C.YELLOW, 'call'),
    'call_processing': ('⚙️', C.YELLOW, 'call'),
    
    # Speech/conversation
    'user_speech': ('🎤', C.CYAN, 'speech'),
    'assistant_response': ('🤖', C.MAGENTA, 'speech'),
    'assistant_ack': ('💬', C.MAGENTA, 'speech'),
    'barge_in': ('✋', C.YELLOW, 'speech'),
    
    # Tool invocations (from conversation)
    'tool_call': ('🔧', C.WHITE, 'tool'),
    'timer_set': ('⏰', C.YELLOW, 'tool'),
    'callback_scheduled': ('📲', C.BLUE, 'tool'),
    'weather_fetch': ('🌤️', C.CYAN, 'tool'),
    
    # Timer events
    'timer_fired': ('🔔', C.YELLOW, 'timer'),
    'timer_complete': ('✅', C.GREEN, 'timer'),
    'timer_cancelled': ('❌', C.RED, 'timer'),
    
    # Callback events
    'callback_execute': ('📲', C.BLUE, 'callback'),
    'callback_dialing': ('📞', C.YELLOW, 'callback'),
    'callback_answered': ('📞', C.GREEN, 'callback'),
    'callback_complete': ('✅', C.GREEN, 'callback'),
    'callback_failed': ('❌', C.RED, 'callback'),
    
    # Task execution (generic)
    'task_scheduled': ('📋', C.BLUE, 'task'),
    'task_execute': ('⚡', C.WHITE, 'task'),
    'task_complete': ('✅', C.GREEN, 'task'),
    'task_cancelled': ('❌', C.RED, 'task'),
    
    # API/Webhook tool calls
    'api_tool_execute': ('🌐', C.BLUE, 'api'),
    'api_tool_complete': ('✅', C.GREEN, 'api'),
    'api_tool_call': ('📤', C.BLUE, 'api'),
    'api_tool_call_initiated': ('📞', C.GREEN, 'api'),
    'api_tool_call_tool_failed': ('❌', C.RED, 'api'),
    'api_tool_no_call': ('⚠️', C.YELLOW, 'api'),
    'api_speak_success': ('🔊', C.GREEN, 'api'),
}

# Events that start a grouped section
GROUP_START_EVENTS = {
    'timer_fired': ('timer', '⏰', 'TIMER FIRED', C.YELLOW),
    'callback_execute': ('callback', '📲', 'CALLBACK', C.BLUE),
    'api_tool_call': ('api', '🌐', 'API TOOL CALL', C.CYAN),
    'api_tool_execute': ('api', '🌐', 'API TOOL EXECUTE', C.CYAN),
    'outbound_call_initiated': ('outbound', '📤', 'OUTBOUND CALL', C.BLUE),
}

# Events that end a grouped section
GROUP_END_EVENTS = {
    'timer': ['timer_complete', 'timer_cancelled', 'call_end'],
    'callback': ['callback_complete', 'callback_failed', 'call_end'],
    'api': ['api_tool_complete', 'api_tool_call_tool_failed', 'outbound_call_initiated'],
    'outbound': ['outbound_call_webhook_success', 'outbound_call_complete', 'outbound_call_failed', 'outbound_call_no_answer'],
}

class CallTracker:
    """Track calls and format output with grouping."""
    
    def __init__(self):
        self.in_call = False
        self.current_caller = None
        self.call_count = 0
        # Track active groups (timer, callback, api, outbound)
        self.active_group = None
        self.group_count = {'timer': 0, 'callback': 0, 'api': 0, 'outbound': 0}
        
    def print_call_header(self, caller: str, direction: str = "inbound"):
        """Print a call start header."""
        self.call_count += 1
        self.in_call = True
        self.current_caller = caller
        
        arrow = "⬅️" if direction == "inbound" else "➡️"
        print(f"\n{C.GREEN}{'━' * 70}{C.NC}")
        print(f"{C.GREEN}  {arrow} CALL #{self.call_count}: {caller}{C.NC}")
        print(f"{C.GREEN}{'━' * 70}{C.NC}\n")
        
    def print_call_footer(self):
        """Print a call end footer."""
        if self.in_call:
            print(f"\n{C.RED}{'─' * 70}{C.NC}")
            print(f"{C.RED}  ✗ CALL ENDED{C.NC}")
            print(f"{C.RED}{'─' * 70}{C.NC}\n")
        self.in_call = False
        self.current_caller = None
        self.end_group()  # End any active group when call ends
    
    def start_group(self, group_type: str, icon: str, title: str, color: str, extra_info: str = ""):
        """Start a grouped section (timer, callback, api)."""
        # End any existing group first
        self.end_group()
        
        self.active_group = group_type
        self.group_count[group_type] = self.group_count.get(group_type, 0) + 1
        count = self.group_count[group_type]
        
        info_str = f" - {extra_info}" if extra_info else ""
        print(f"\n{color}{'┌' + '─' * 50}{C.NC}")
        print(f"{color}│ {icon} {title} #{count}{info_str}{C.NC}")
        print(f"{color}{'└' + '─' * 50}{C.NC}")
    
    def end_group(self):
        """End the current grouped section."""
        if self.active_group:
            self.active_group = None

tracker = CallTracker()

def format_log(line: str, show_all: bool = False) -> str | None:
    """Format a log line for display. Returns None to skip."""
    global tracker
    
    line = line.strip()
    if not line:
        return None
    
    # Handle docker compose prefix (service-name  | log)
    if ' | ' in line:
        parts = line.split(' | ', 1)
        if len(parts) == 2:
            line = parts[1]
    
    # Try to parse JSON
    try:
        start = line.find('{')
        if start == -1:
            # Not JSON
            if show_all:
                return f"{C.GRAY}{line}{C.NC}"
            lower = line.lower()
            if 'error' in lower:
                return f"{C.RED}  ❌ {line}{C.NC}"
            elif 'warning' in lower or 'warn' in lower:
                return f"{C.YELLOW}  ⚠️  {line}{C.NC}"
            return None
        
        data = json.loads(line[start:])
        
        event = data.get('event')
        level = data.get('level', 'INFO')
        msg = data.get('msg', '')
        ts = data.get('ts', '')
        extra = data.get('data', {})
        
        # Extract time (HH:MM:SS)
        time_str = ts
        if ' ' in ts:
            time_str = ts.split(' ')[-1]
            if ',' in time_str:
                time_str = time_str.split(',')[0]
        
        # Handle call lifecycle events specially
        if event == 'call_start':
            caller = extra.get('caller', msg)
            direction = extra.get('direction', 'inbound')
            tracker.print_call_header(caller, direction)
            return None  # Header already printed
            
        if event == 'call_end':
            tracker.print_call_footer()
            return None  # Footer already printed
        
        # Handle group start events (timer_fired, callback_execute, api_tool_call, outbound_call_initiated)
        if event in GROUP_START_EVENTS:
            group_type, icon, title, color = GROUP_START_EVENTS[event]
            # Build extra info from event data
            extra_info = ""
            if event == 'timer_fired':
                extra_info = extra.get('message', msg) if extra else msg
            elif event == 'callback_execute':
                extra_info = extra.get('destination', extra.get('target', ''))
            elif event in ('api_tool_call', 'api_tool_execute'):
                tool = extra.get('tool', '')
                ext = extra.get('extension', '')
                extra_info = f"{tool}" + (f" → {ext}" if ext else "")
            elif event == 'outbound_call_initiated':
                ext = extra.get('extension', extra.get('uri', ''))
                call_id = extra.get('call_id', '')
                extra_info = ext
                if call_id:
                    extra_info += f" ({call_id})"
            
            tracker.start_group(group_type, icon, title, color, extra_info)
            # Continue to also print the event line below
        
        # Handle group end events
        if tracker.active_group:
            end_events = GROUP_END_EVENTS.get(tracker.active_group, [])
            if event in end_events:
                tracker.end_group()
        
        # Format based on event
        if event and event in EVENT_STYLE:
            icon, color, category = EVENT_STYLE[event]
            
            # Indent based on category and active group
            indent = "  "
            if tracker.active_group:
                indent = "    "  # Indent within group
            elif category == 'speech':
                indent = "    "  # Extra indent for conversation
            elif category in ('tool', 'task', 'timer', 'callback', 'api', 'outbound'):
                indent = "      "  # Extra indent for tools/tasks
            
            # Build output line
            output = f"{C.GRAY}{time_str}{C.NC} {indent}{icon} {color}{msg}{C.NC}"
            
            # Add relevant extra data
            if extra:
                # Filter out redundant info already in message
                show_keys = []
                if event == 'user_speech':
                    pass  # Text already in msg
                elif event == 'tool_call':
                    show_keys = ['params']
                elif event == 'task_scheduled':
                    show_keys = ['delay', 'target']
                elif event in ('timer_set', 'timer_fired'):
                    show_keys = ['duration', 'message']
                elif event == 'callback_scheduled':
                    show_keys = ['delay', 'destination']
                elif event in ('api_tool_call', 'api_tool_execute', 'api_tool_call_initiated'):
                    show_keys = ['tool', 'extension', 'params']
                elif event == 'weather_fetch':
                    pass  # Message has the summary
                elif event == 'outbound_call_initiated':
                    show_keys = ['extension', 'call_id']
                elif event == 'outbound_call_choice_collected':
                    show_keys = ['choice', 'raw_text']
                elif event.startswith('outbound_call_'):
                    show_keys = ['call_id', 'status', 'duration']
                else:
                    show_keys = list(extra.keys())
                
                if show_keys:
                    extra_parts = []
                    for k in show_keys:
                        if k in extra and extra[k]:
                            v = extra[k]
                            if isinstance(v, dict):
                                v = ', '.join(f"{kk}={vv}" for kk, vv in v.items())
                            extra_parts.append(f"{k}={v}")
                    if extra_parts:
                        output += f" {C.GRAY}({', '.join(extra_parts)}){C.NC}"
            
            return output
        
        # Show errors/warnings
        if level in ('ERROR', 'CRITICAL'):
            return f"{C.GRAY}{time_str}{C.NC}   {C.RED}❌ {msg}{C.NC}"
        if level == 'WARNING':
            return f"{C.GRAY}{time_str}{C.NC}   {C.YELLOW}⚠️  {msg}{C.NC}"
        
        # Show all other logs if -a flag
        if show_all:
            return f"{C.GRAY}{time_str}   {msg}{C.NC}"
        
        return None
        
    except json.JSONDecodeError:
        if show_all:
            return f"{C.GRAY}{line}{C.NC}"
        lower = line.lower()
        if 'error' in lower:
            return f"{C.RED}  ❌ {line}{C.NC}"
        elif 'warning' in lower or 'warn' in lower:
            return f"{C.YELLOW}  ⚠️  {line}{C.NC}"
        return None

def print_header(show_all: bool):
    print(f"{C.WHITE}{'═' * 70}{C.NC}")
    print(f"{C.WHITE} 🤖📞 SIP AI Assistant - CHAOS.CORP{C.NC}")
    print(f"{C.WHITE}{'═' * 70}{C.NC}")
    # if show_all:
    #     print(f"{C.GRAY}  Mode: ALL logs{C.NC}")
    # else:
    #     print(f"{C.GRAY}  Showing: calls, speech, tools, tasks, errors{C.NC}")
    #     print(f"{C.GRAY}  Use -a to show all logs{C.NC}")
    # print(f"{C.WHITE}{'─' * 70}{C.NC}")
    # print()

def process_stream(stream, show_all: bool):
    for line in stream:
        formatted = format_log(line, show_all)
        if formatted:
            print(formatted, flush=True)

def main():
    parser = argparse.ArgumentParser(description='View SIP AI Assistant logs')
    parser.add_argument('container', nargs='?', default='sip-agent',
                       help='Service name (default: sip-agent)')
    parser.add_argument('--stdin', action='store_true',
                       help='Read from stdin instead of docker compose logs')
    parser.add_argument('-a', '--all', action='store_true',
                       help='Show all logs, not just interesting events')
    parser.add_argument('--no-header', action='store_true',
                       help='Skip the header')
    args = parser.parse_args()
    
    if not args.no_header:
        print_header(args.all)
    
    try:
        if args.stdin:
            process_stream(sys.stdin, args.all)
        else:
            cmd = ['docker',  'logs', '--tail', '0', '-f', args.container]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors='replace'
            )
            process_stream(process.stdout, args.all)
            
    except KeyboardInterrupt:
        print(f"\n{C.GRAY}Stopped.{C.NC}")
        sys.exit(0)
    except BrokenPipeError:
        sys.exit(0)
    except FileNotFoundError:
        print(f"{C.RED}Error: docker not found. Use --stdin to pipe logs.{C.NC}")
        sys.exit(1)

if __name__ == '__main__':
    main()