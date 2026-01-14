ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG CUDA_FLAVOR=cudnn-devel

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python 3.12 en Ubuntu 24.04 (repos oficiales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    git curl ca-certificates \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Entorno virtual para evitar PEP 668
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /comfyui
COPY . .

# PyTorch CUDA 12.8 (cu128)
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PIP_INDEX_URL=${PYTORCH_INDEX_URL}
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
       torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    && python -m pip install --no-cache-dir -r requirements.txt

# (Opcional pero recomendado) Instalar flash-attn desde wheel prebuilt (evita compilación)
# Ojo: usa el wheel que corresponda EXACTO a tu combinación cp312/torch2.9/cu12/abiTRUE
RUN python -m pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
