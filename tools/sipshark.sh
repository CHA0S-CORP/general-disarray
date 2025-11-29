#!/bin/bash

# ==============================================================================
# SIP-SHARK: A wrapper for termshark to focus purely on SIP traffic.
# ==============================================================================

# Default settings
INTERFACE="any"
CAPTURE_FILTER="port 5060 or port 5061"
DISPLAY_FILTER="sip"
OUTPUT_FILE=""

# Help Function
show_help() {
    echo "Usage: ./sipshark.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i <interface>   Specify network interface (default: any)"
    echo "  -w <file.pcap>   Write captured packets to a file"
    echo "  -r               Include RTP traffic (VoIP audio) in capture"
    echo "  -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./sipshark.sh -i eth0"
    echo "  ./sipshark.sh -w capture.pcap"
    echo "  ./sipshark.sh -i lo -r"
    echo ""
}

# Check if termshark is installed
if ! command -v termshark &> /dev/null; then
    echo "Error: termshark is not installed or not in your PATH."
    echo "Please install it via your package manager (e.g., 'apt install termshark' or 'brew install termshark')."
    exit 1
fi

# Parse Arguments
while getopts "i:w:rh" opt; do
    case ${opt} in
        i)
            INTERFACE=$OPTARG
            ;;
        w)
            OUTPUT_FILE="-w $OPTARG"
            ;;
        r)
            # Add UDP range for RTP if requested (Common RTP ports, adjust as needed)
            # Standard RTP often floats between 10000-20000, but we'll cast a wider net or rely on decoding.
            # For simplicity in a capture filter, we often just add udp.
            echo "Including RTP in capture filter..."
            CAPTURE_FILTER="port 5060 or port 5061 or udp"
            # We update display filter to show SIP or RTP
            DISPLAY_FILTER="sip || rtp"
            ;;
        h)
            show_help
            exit 0
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Starting SIP Capture on: $INTERFACE"
echo "Capture Filter (BPF):    $CAPTURE_FILTER"
echo "Display Filter:          $DISPLAY_FILTER"
if [ ! -z "$OUTPUT_FILE" ]; then
    echo "Saving to:               ${OUTPUT_FILE:3}"
fi
echo "========================================"
echo "Press 'Ctrl+C' to stop capturing."
echo ""

# Execute termshark
# -i: Interface
# -f: Capture filter (Berkeley Packet Filter syntax) - limits what is saved/processed
# -Y: Display filter (Wireshark syntax) - limits what is shown in the packet list
termshark -i "$INTERFACE" -f "$CAPTURE_FILTER" -Y "$DISPLAY_FILTER" $OUTPUT_FILE