#!/usr/bin/env python3
"""
SIP AI Assistant - Log Viewer
Filters and displays interesting events from JSON logs.

Usage:
    ./view-logs.py                          # Tail docker compose logs
    ./view-logs.py sip-ai-assistant         # Tail specific container
    docker logs -f container | ./view-logs.py --stdin
"""

import sys
import json
import argparse
import subprocess
from datetime import datetime

# ANSI colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    GRAY = '\033[0;90m'
    NC = '\033[0m'  # No Color

# Events to show
INTERESTING_EVENTS = {
    # Call events
    'call_start': ('📞', Colors.GREEN),
    'call_end': ('📴', Colors.RED),
    'call_timeout': ('📴', Colors.RED),
    
    # Speech events
    'user_speech': ('🎤', Colors.CYAN),
    'assistant_response': ('🤖', Colors.MAGENTA),
    'assistant_ack': ('🤖', Colors.MAGENTA),
    
    # Timer/Callback events
    'timer_set': ('⏰', Colors.YELLOW),
    'timer_fired': ('⏰', Colors.YELLOW),
    'callback_scheduled': ('📲', Colors.BLUE),
    'callback_execute': ('📲', Colors.BLUE),
    'callback_complete': ('📲', Colors.BLUE),
    
    # Tool events
    'tool_call': ('🔧', Colors.WHITE),
    'task_execute': ('🔧', Colors.WHITE),
    
    # Other
    'barge_in': ('✋', Colors.YELLOW),
}

def format_log(line: str) -> str | None:
    """Format a log line for display. Returns None to skip."""
    line = line.strip()
    if not line:
        return None
    
    # Handle docker compose prefix (container name | log)
    if ' | ' in line:
        parts = line.split(' | ', 1)
        if len(parts) == 2:
            line = parts[1]
    
    # Try to parse JSON
    try:
        # Find JSON object in line
        start = line.find('{')
        if start == -1:
            # Not JSON - check for errors/warnings
            lower = line.lower()
            if 'error' in lower:
                return f"{Colors.RED}{line}{Colors.NC}"
            elif 'warning' in lower or 'warn' in lower:
                return f"{Colors.YELLOW}{line}{Colors.NC}"
            return None
        
        data = json.loads(line[start:])
        
        event = data.get('event')
        level = data.get('level', 'INFO')
        msg = data.get('msg', '')
        ts = data.get('ts', '')
        extra_data = data.get('data', {})
        
        # Check if interesting event
        if event and event in INTERESTING_EVENTS:
            icon, color = INTERESTING_EVENTS[event]
            
            # Extract time portion
            time_str = ts
            if 'T' in ts:
                time_str = ts.split('T')[1].split('.')[0] if '.' in ts.split('T')[1] else ts.split('T')[1]
            elif ',' in ts:
                time_str = ts.split(',')[0].split(' ')[-1]
            
            # Format output
            output = f"{Colors.GRAY}{time_str}{Colors.NC} {color}{icon} [{event:20}]{Colors.NC} {msg}"
            
            # Add extra data if present
            if extra_data:
                data_str = ', '.join(f"{k}={v}" for k, v in extra_data.items())
                output += f" {Colors.GRAY}({data_str}){Colors.NC}"
            
            return output
        
        # Check for errors/warnings
        if level in ('ERROR', 'CRITICAL'):
            return f"{Colors.RED}❌ {ts} [{level}] {msg}{Colors.NC}"
        elif level == 'WARNING':
            return f"{Colors.YELLOW}⚠️  {ts} [{level}] {msg}{Colors.NC}"
        
        return None
        
    except json.JSONDecodeError:
        # Not valid JSON - check for errors/warnings in plain text
        lower = line.lower()
        if 'error' in lower:
            return f"{Colors.RED}{line}{Colors.NC}"
        elif 'warning' in lower or 'warn' in lower:
            return f"{Colors.YELLOW}{line}{Colors.NC}"
        return None

def print_header():
    """Print the header."""
    print(f"{Colors.WHITE}═══════════════════════════════════════════════════════════════{Colors.NC}")
    print(f"{Colors.WHITE}  SIP AI Assistant - Filtered Event Log{Colors.NC}")
    print(f"{Colors.WHITE}═══════════════════════════════════════════════════════════════{Colors.NC}")
    print(f"{Colors.GRAY}Events: {', '.join(INTERESTING_EVENTS.keys())}{Colors.NC}")
    print(f"{Colors.GRAY}Plus: errors and warnings{Colors.NC}")
    print(f"{Colors.WHITE}───────────────────────────────────────────────────────────────{Colors.NC}")
    print()

def process_stream(stream):
    """Process a stream of log lines."""
    for line in stream:
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='replace')
        
        formatted = format_log(line)
        if formatted:
            print(formatted, flush=True)

def main():
    parser = argparse.ArgumentParser(description='View SIP AI Assistant logs')
    parser.add_argument('container', nargs='?', default='sip-ai-assistant',
                       help='Container name (default: sip-ai-assistant)')
    parser.add_argument('--stdin', action='store_true',
                       help='Read from stdin instead of docker logs')
    parser.add_argument('--no-header', action='store_true',
                       help='Skip the header')
    args = parser.parse_args()
    
    if not args.no_header:
        print_header()
    
    try:
        if args.stdin:
            process_stream(sys.stdin)
        else:
            # Run docker logs -f
            cmd = ['docker', 'logs', '-f', args.container]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1
            )
            process_stream(process.stdout)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.GRAY}Stopped.{Colors.NC}")
        sys.exit(0)
    except BrokenPipeError:
        sys.exit(0)

if __name__ == '__main__':
    main()
