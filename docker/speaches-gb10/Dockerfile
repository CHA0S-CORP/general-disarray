# speaches for DGX Spark / GB10 (aarch64, Blackwell sm_121)
# ==========================================================
# The stock speaches image bundles the PyPI ctranslate2 wheel, which is
# CPU-only on linux/aarch64 — Whisper silently runs on the Grace cores at
# ~1.1x realtime (speaches-ai/speaches#620). This derived image rebuilds
# CTranslate2 from source with CUDA 13 (the first toolkit that knows GB10's
# sm_121) and layers the wheel + the CUDA 13 runtime libs it needs onto the
# stock image.
#
# Build (from the repo root; ~20-40 min, compiles CUDA kernels):
#   docker build -t ghcr.io/speaches-ai/speaches:gb10-cuda13 docker/speaches-gb10
#
# The compose files point the speaches service at the gb10-cuda13 tag.

ARG CT2_VERSION=v4.6.2

# --- Stage 1: build CTranslate2 with CUDA 13 for sm_121 ---------------------
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake ninja-build g++ python3 python3-dev python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG CT2_VERSION
RUN git clone --depth 1 --branch ${CT2_VERSION} --recursive \
        https://github.com/OpenNMT/CTranslate2.git /src/CTranslate2

# WITH_RUY: bundled CPU GEMM, no runtime library dependency (the official
# aarch64 wheels use it too). CT2 uses the legacy FindCUDA path, so the arch
# knob is CUDA_ARCH_LIST (not CMAKE_CUDA_ARCHITECTURES); its "Auto" default
# expands to ancient archs that CUDA 13's nvcc rejects (compute_53), and the
# cmake 3.28 arch table predates Blackwell so "12.1" is not a valid name
# there. GB10's sm_121 SASS is therefore injected via CUDA_NVCC_FLAGS (CT2
# appends the selected arch flags to it); 9.0+PTX stays as the JIT fallback
# for other Grace-class GPUs.
RUN cmake -S /src/CTranslate2 -B /build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/ctranslate2 \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF -DWITH_RUY=ON \
        -DOPENMP_RUNTIME=COMP \
        -DBUILD_CLI=OFF \
        -DCUDA_ARCH_LIST="9.0+PTX" \
        -DCUDA_NVCC_FLAGS="-gencode=arch=compute_121,code=sm_121" \
    && cmake --build /build --target install -j"$(nproc)"

# Python wheel (cp312 — matches both ubuntu24.04 and the speaches venv).
RUN python3 -m venv /buildvenv \
    && /buildvenv/bin/pip install --no-cache-dir pybind11 setuptools wheel \
    && cd /src/CTranslate2/python \
    && CTRANSLATE2_ROOT=/opt/ctranslate2 /buildvenv/bin/pip wheel . --no-deps -w /wheels

# --- Stage 2: overlay onto the stock speaches image --------------------------
FROM ghcr.io/speaches-ai/speaches:0.9.0-rc.2-cuda-12.6.3

USER root
# libctranslate2.so + the CUDA 13 runtime it links against. Kept in a private
# prefix so the image's own CUDA 12.6 libs are untouched; LD_LIBRARY_PATH
# resolves the .so.13 sonames only for consumers that ask for them (cudnn .so.9
# is also taken from here — CTranslate2 is this image's only cudnn consumer).
COPY --from=builder /opt/ctranslate2/lib/ /opt/ctranslate2/lib/
COPY --from=builder /usr/local/cuda/lib64/libcudart.so.13* /opt/ctranslate2/lib/
COPY --from=builder /usr/local/cuda/lib64/libcublas.so.13* /opt/ctranslate2/lib/
COPY --from=builder /usr/local/cuda/lib64/libcublasLt.so.13* /opt/ctranslate2/lib/
COPY --from=builder /usr/local/cuda/lib64/libnvJitLink.so.13* /opt/ctranslate2/lib/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libcudnn*.so.9* /opt/ctranslate2/lib/
COPY --from=builder /wheels /tmp/wheels
RUN chown -R ubuntu:ubuntu /tmp/wheels

USER ubuntu
RUN uv pip install --python /home/ubuntu/speaches/.venv/bin/python \
        --force-reinstall --no-deps /tmp/wheels/ctranslate2-*.whl

USER root
RUN rm -rf /tmp/wheels
USER ubuntu
ENV LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH}
