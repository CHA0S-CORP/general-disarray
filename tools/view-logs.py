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
# Categories: 'system', 'call', 'speech', 'tool', 'task'
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
    'outbound_call_initiated': ('📤', C.BLUE, 'call'),
    'outbound_call_dialing': ('📞', C.YELLOW, 'call'),
    'outbound_call_answered': ('📞', C.GREEN, 'call'),
    'outbound_call_no_answer': ('📴', C.RED, 'call'),
    'outbound_call_message_played': ('🔊', C.GREEN, 'call'),
    'outbound_call_ack_played': ('✅', C.GREEN, 'call'),
    'outbound_call_choice_collected': ('✅', C.GREEN, 'call'),
    'outbound_call_choice_no_match': ('❓', C.YELLOW, 'call'),
    'outbound_call_webhook': ('🔗', C.BLUE, 'call'),
    'outbound_call_webhook_success': ('✅', C.GREEN, 'call'),
    
    # Call queue
    'call_queued': ('📥', C.BLUE, 'call'),
    'call_waited': ('⏳', C.YELLOW, 'call'),
    'call_processing': ('⚙️', C.YELLOW, 'call'),
    
    # Speech/conversation
    'user_speech': ('🎤', C.CYAN, 'speech'),
    'assistant_response': ('🤖', C.MAGENTA, 'speech'),
    'assistant_ack': ('💬', C.MAGENTA, 'speech'),
    'barge_in': ('✋', C.YELLOW, 'speech'),
    
    # Tool invocations
    'tool_call': ('🔧', C.WHITE, 'tool'),
    'timer_set': ('⏰', C.YELLOW, 'tool'),
    'callback_scheduled': ('📲', C.BLUE, 'tool'),
    'weather_fetch': ('🌤️', C.CYAN, 'tool'),
    
    # Task execution
    'task_scheduled': ('📋', C.BLUE, 'task'),
    'task_execute': ('⚡', C.WHITE, 'task'),
    'timer_fired': ('🔔', C.YELLOW, 'task'),
    'callback_execute': ('📲', C.BLUE, 'task'),
    'callback_complete': ('✅', C.GREEN, 'task'),
}

class CallTracker:
    """Track calls and format output with grouping."""
    
    def __init__(self):
        self.in_call = False
        self.current_caller = None
        self.call_count = 0
        
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
        
        # Format based on event
        if event and event in EVENT_STYLE:
            icon, color, category = EVENT_STYLE[event]
            
            # Indent based on category
            indent = "  "
            if category == 'speech':
                indent = "    "  # Extra indent for conversation
            elif category in ('tool', 'task'):
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
