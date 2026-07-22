FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-venv \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip "setuptools==69.5.1" uv \
    && git lfs install --system

ENV GWM_PATH=/workspace
ENV UV_LINK_MODE=copy
ENV UV_SYSTEM_PYTHON=1
ENV TORCH_CUDA_ARCH_LIST=7.5
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

ENTRYPOINT ["bash", "/workspace/scripts/container_entrypoint.sh"]
CMD ["sleep", "infinity"]
