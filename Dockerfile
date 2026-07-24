FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_BREAK_SYSTEM_PACKAGES=1

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
        python3-pip \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --break-system-packages uv \
    && uv python install 3.10 \
    && ln -sf "$(uv python find --python-preference only-managed 3.10)" /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python \
    && uv pip install --system --python /usr/local/bin/python3 \
        "pip>=24" \
        "setuptools==69.5.1" \
        wheel \
    && ln -sf "$(dirname "$(uv python find --python-preference only-managed 3.10)")/pip" /usr/local/bin/pip \
    && ln -sf "$(dirname "$(uv python find --python-preference only-managed 3.10)")/pip3" /usr/local/bin/pip3 \
    && git lfs install --system

ENV GWM_PATH=/workspace
ENV UV_LINK_MODE=copy
ENV UV_SYSTEM_PYTHON=1
ENV UV_PYTHON=/usr/local/bin/python3
ENV PYTHON_BIN=/usr/local/bin/python3
ENV PATH=/usr/local/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST=auto
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

ENTRYPOINT ["bash", "/workspace/scripts/container_entrypoint.sh"]
CMD ["sleep", "infinity"]
