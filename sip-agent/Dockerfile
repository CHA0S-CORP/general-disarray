# ============================================================================
# SIP AI Assistant - Lightweight Container
# ============================================================================
# All ML inference is offloaded to dedicated API services:
# - Speaches API for STT (Whisper) and TTS (Piper/Kokoro)
# - vLLM for LLM
#
# This container only needs:
# - PJSIP for SIP connectivity
# - Audio processing libraries
# - HTTP clients for API communication

# ============================================================================
# STAGE 1: Builder - Build PJSIP Python bindings
# ============================================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    tar \
    pkg-config \
    python3-dev \
    python3-pip \
    # Audio dev libs for PJSIP
    libasound2-dev \
    libssl-dev \
    libopus-dev \
    libv4l-dev \
    libsdl2-dev \
    libportaudio2 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Build SWIG 4.0.2 (needed for PJSIP Python bindings)
WORKDIR /tmp
RUN wget -q http://prdownloads.sourceforge.net/swig/swig-4.0.2.tar.gz && \
    tar -xf swig-4.0.2.tar.gz && \
    cd swig-4.0.2 && \
    wget -q https://sourceforge.net/projects/pcre/files/pcre/8.45/pcre-8.45.tar.gz && \
    ./Tools/pcre-build.sh && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf swig-4.0.2*

# Build PJSIP (STATIC with fPIC)
WORKDIR /tmp
RUN wget -q https://github.com/pjsip/pjproject/archive/refs/tags/2.14.tar.gz && \
    tar xzf 2.14.tar.gz && \
    cd pjproject-2.14 && \
    export CFLAGS="$CFLAGS -fPIC" && \
    export CXXFLAGS="$CXXFLAGS -fPIC" && \
    ./configure --with-opus && \
    make dep && \
    make -j$(nproc) && \
    make install

# Build Python Wheel
RUN cd /tmp/pjproject-2.14/pjsip-apps/src/swig && \
    make python && \
    cd python && \
    python3 setup.py bdist_wheel && \
    mkdir -p /build_artifacts && \
    cp dist/*.whl /build_artifacts/


# ============================================================================
# STAGE 2: Final Runtime Image (Lightweight)
# ============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    # Runtime audio libs
    libasound2t64 \
    libssl3t64 \
    libopus0 \
    libv4l-0 \
    libsdl2-2.0-0 \
    libportaudio2 \
    libsndfile1 \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PJSIP wheel from builder
COPY --from=builder /build_artifacts/*.whl /tmp/
RUN pip install /tmp/*.whl --break-system-packages && rm /tmp/*.whl

# Install Python dependencies (lightweight - no ML frameworks)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy application code
COPY src/*.py ./

# Create directories
RUN mkdir -p /app/data/recordings /app/data/logs /app/data/voices

# Expose ports
EXPOSE 5060/udp 5060/tcp 10000-10100/udp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Run
CMD ["python3", "main.py"]
