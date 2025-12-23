#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ComfyUI

mkdir -p /comfyui/{custom_nodes,input,output,models,temp}

export COMFY_ALLOW_DOWNLOAD=1
export COMFY_ALLOW_INSTALL=1
export COMFY_INSTALL_TOKEN=tu_token_real
export COMFY_ALLOW_GIT_UPDATE=1
export COMFY_GIT_UPSTREAM_URL=https://github.com/comfyanonymous/ComfyUI.git
export COMFY_GIT_ORIGIN_URL=https://github.com/BeckZero/ComfyUI.git
export COMFY_GIT_BRANCH=master

python3 -m pip install -r requirements.txt

 


COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"

if [[ ! -d "$COMFYUI_DIR" ]]; then
  echo "ComfyUI dir not found: $COMFYUI_DIR" >&2
  exit 1
fi

cd "$COMFYUI_DIR"

mkdir -p models/vae models/text_encoders models/diffusion_models custom_nodes

download() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    echo "skip (exists): $out"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$out" "$url"
  else
    echo "wget or curl is required to download files" >&2
    exit 1
  fi
}

download "https://huggingface.co/gguf-org/z-image-gguf/resolve/main/pig_flux_vae_fp32-f16.gguf?download=true" \
  "./models/vae/pig_flux_vae_fp32-f16.gguf"

download "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q5_K_M.gguf?download=true" \
  "./models/text_encoders/Qwen3-4B-Q5_K_M.gguf"

download "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q5_K_M.gguf?download=true" \
  "./models/diffusion_models/z_image_turbo-Q5_K_M.gguf"

if [[ -d "custom_nodes/gguf/.git" ]]; then
  git -C custom_nodes/gguf pull --ff-only
else
  git clone https://github.com/calcuis/gguf custom_nodes/gguf
fi

apt-get update && apt-get install -y lsof


# Solo si quieres que el script arranque ComfyUI:
python3 main.py --listen 0.0.0.0 --port 8188