#!/usr/bin/env python3
"""
SIP AI Assistant - Log Viewer
Filters and displays interesting events from JSON logs.

Usage:
    ./view-logs.py                          # Tail docker compose logs
    ./view-logs.py sip-ai-assistant         # Tail specific container  
    ./view-logs.py -a                       # Show ALL logs (not just events)
    docker logs -f container | ./view-logs.py --stdin
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

# Event styling: event_name -> (icon, color)
EVENT_STYLE = {
    # Startup
    'warming_up': ('🔥', C.YELLOW),
    'ready': ('✅', C.GREEN),
    
    # Call events
    'call_start': ('📞', C.GREEN),
    'call_end': ('📴', C.RED),
    'call_timeout': ('📴', C.RED),
    
    # Speech events
    'user_speech': ('🎤', C.CYAN),
    'assistant_response': ('🤖', C.MAGENTA),
    'assistant_ack': ('💬', C.MAGENTA),
    
    # Timer/Callback events
    'timer_set': ('⏰', C.YELLOW),
    'timer_fired': ('🔔', C.YELLOW),
    'callback_scheduled': ('📲', C.BLUE),
    'callback_execute': ('📲', C.BLUE),
    'callback_complete': ('✅', C.GREEN),
    
    # Tool events
    'tool_call': ('🔧', C.WHITE),
    'task_execute': ('⚡', C.WHITE),
    
    # Other
    'barge_in': ('✋', C.YELLOW),
}

def format_log(line: str, show_all: bool = False) -> str | None:
    """Format a log line for display. Returns None to skip."""
    line = line.strip()
    if not line:
        return None
    
    # Handle docker compose prefix (container name | log)
    container = None
    if ' | ' in line:
        parts = line.split(' | ', 1)
        if len(parts) == 2:
            container = parts[0].strip()
            line = parts[1]
    
    # Skip non-assistant containers unless showing all
    if container and 'assistant' not in container.lower() and not show_all:
        return None
    
    # Try to parse JSON
    try:
        start = line.find('{')
        if start == -1:
            # Not JSON
            if show_all:
                return f"{C.GRAY}{line}{C.NC}"
            # Check for errors/warnings in plain text
            lower = line.lower()
            if 'error' in lower:
                return f"{C.RED}❌ {line}{C.NC}"
            elif 'warning' in lower or 'warn' in lower:
                return f"{C.YELLOW}⚠️  {line}{C.NC}"
            return None
        
        data = json.loads(line[start:])
        
        event = data.get('event')
        level = data.get('level', 'INFO')
        msg = data.get('msg', '')
        ts = data.get('ts', '')
        extra = data.get('data', {})
        
        # Extract time
        time_str = ts
        if ' ' in ts:
            time_str = ts.split(' ')[-1]
            if ',' in time_str:
                time_str = time_str.split(',')[0]
        
        # Format based on event
        if event and event in EVENT_STYLE:
            icon, color = EVENT_STYLE[event]
            output = f"{C.GRAY}{time_str}{C.NC} {icon} {color}[{event}]{C.NC} {msg}"
            if extra:
                extra_str = ' '.join(f"{k}={v}" for k, v in extra.items())
                output += f" {C.GRAY}({extra_str}){C.NC}"
            return output
        
        # Show errors/warnings
        if level in ('ERROR', 'CRITICAL'):
            return f"{C.GRAY}{time_str}{C.NC} {C.RED}❌ [{level}]{C.NC} {msg}"
        if level == 'WARNING':
            return f"{C.GRAY}{time_str}{C.NC} {C.YELLOW}⚠️  [{level}]{C.NC} {msg}"
        
        # Show all other logs if -a flag
        if show_all:
            return f"{C.GRAY}{time_str} [{level}] {msg}{C.NC}"
        
        return None
        
    except json.JSONDecodeError:
        if show_all:
            return f"{C.GRAY}{line}{C.NC}"
        lower = line.lower()
        if 'error' in lower:
            return f"{C.RED}❌ {line}{C.NC}"
        elif 'warning' in lower or 'warn' in lower:
            return f"{C.YELLOW}⚠️  {line}{C.NC}"
        return None

def print_header(show_all: bool):
    events = ', '.join(EVENT_STYLE.keys())
    print(f"{C.WHITE}{'═' * 70}{C.NC}")
    print(f"{C.WHITE}  SIP AI Assistant - Log Viewer{C.NC}")
    print(f"{C.WHITE}{'═' * 70}{C.NC}")
    if show_all:
        print(f"{C.GRAY}Mode: Showing ALL logs{C.NC}")
    else:
        print(f"{C.GRAY}Events: {events}{C.NC}")
        print(f"{C.GRAY}Use -a to show all logs{C.NC}")
    print(f"{C.WHITE}{'─' * 70}{C.NC}")
    print()

def process_stream(stream, show_all: bool):
    for line in stream:
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='replace')
        formatted = format_log(line, show_all)
        if formatted:
            print(formatted, flush=True)

def main():
    parser = argparse.ArgumentParser(description='View SIP AI Assistant logs')
    parser.add_argument('container', nargs='?', default='sip-agent',
                       help='Container name (default: sip-agent)')
    parser.add_argument('--stdin', action='store_true',
                       help='Read from stdin instead of docker logs')
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
            cmd = ['docker', 'compose', 'logs', '-f', args.container]
            print(f"{C.GRAY}Running: {' '.join(cmd)}{C.NC}\n")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1
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
