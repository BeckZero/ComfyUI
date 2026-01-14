#!/usr/bin/env bash
set -euo pipefail

# cd /workspace/ComfyUI

# mkdir -p /comfyui/{custom_nodes,input,output,models,temp}
mkdir -p /comfyui/{input,output,temp}

export COMFY_ALLOW_DOWNLOAD=1
export COMFY_ALLOW_INSTALL=1
export COMFY_INSTALL_TOKEN=tu_token_real
export COMFY_ALLOW_GIT_UPDATE=1
export COMFY_GIT_UPSTREAM_URL=https://github.com/comfyanonymous/ComfyUI.git
export COMFY_GIT_ORIGIN_URL=https://github.com/BeckZero/ComfyUI.git
export COMFY_GIT_BRANCH=master
export CLOUDFLARE_TUNNEL_ENABLE=1
export CLOUDFLARE_TUNNEL_PROTOCOL=http2

python3 -m pip install -r requirements.txt

 


# COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
COMFYUI_DIR="${COMFYUI_DIR:-/ComfyUI}"

if [[ ! -d "$COMFYUI_DIR" ]]; then
  echo "ComfyUI dir not found: $COMFYUI_DIR" >&2
  exit 1
fi

cd "$COMFYUI_DIR"

# mkdir -p models/vae models/text_encoders models/diffusion_models custom_nodes

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

# Download models in parallel to reduce startup time.
# pids=()
# download "https://huggingface.co/gguf-org/z-image-gguf/resolve/main/pig_flux_vae_fp32-f16.gguf?download=true" \
#   "./models/vae/pig_flux_vae_fp32-f16.gguf" &
# pids+=("$!")

# download "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q5_K_M.gguf?download=true" \
#   "./models/text_encoders/Qwen3-4B-Q5_K_M.gguf" &
# pids+=("$!")

# download "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q5_K_M.gguf?download=true" \
#   "./models/diffusion_models/z_image_turbo-Q5_K_M.gguf" &
# pids+=("$!")

# download "https://civitai.com/api/download/models/2530056?type=Model&format=SafeTensor" \
#   "./models/loras/Mystic-XXX-ZIT-v3.safetensors" &
# pids+=("$!") 

# download_failed=0
# for pid in "${pids[@]}"; do
#   if ! wait "$pid"; then
#     download_failed=1
#   fi
# done
# if ((download_failed)); then
#   echo "one or more model downloads failed" >&2
#   exit 1
# fi

# if [[ -d "custom_nodes/gguf/.git" ]]; then
#   git -C custom_nodes/gguf pull --ff-only
# else
#   git clone https://github.com/calcuis/gguf custom_nodes/gguf
# fi

apt-get update && apt-get install -y lsof



##### cloudflare
CLOUDFLARE_TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN:-}"
CLOUDFLARE_TUNNEL_URL="${CLOUDFLARE_TUNNEL_URL:-http://127.0.0.1:8188}"
CLOUDFLARE_TUNNEL_ENABLE="${CLOUDFLARE_TUNNEL_ENABLE:-0}"
CLOUDFLARE_TUNNEL_PROTOCOL="${CLOUDFLARE_TUNNEL_PROTOCOL:-http2}"

if [[ -n "$CLOUDFLARE_TUNNEL_TOKEN" || "$CLOUDFLARE_TUNNEL_ENABLE" == "1" ]]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    arch="$(uname -m)"
    case "$arch" in
      x86_64|amd64) cf_arch="amd64" ;;
      aarch64|arm64) cf_arch="arm64" ;;
      *) echo "unsupported arch for cloudflared: $arch" >&2; exit 1 ;;
    esac
    bin_dir="/usr/local/bin"
    if [[ ! -w "$bin_dir" ]]; then
      bin_dir="${HOME:-/root}/.local/bin"
      mkdir -p "$bin_dir"
      export PATH="$bin_dir:$PATH"
    fi
    download "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$cf_arch" \
      "${bin_dir}/cloudflared"
    chmod +x "${bin_dir}/cloudflared"
  fi

  if [[ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]]; then
    cloudflared tunnel --no-autoupdate --protocol "$CLOUDFLARE_TUNNEL_PROTOCOL" run \
      --token "$CLOUDFLARE_TUNNEL_TOKEN" \
      > /var/log/cloudflared.log 2>&1 &
    echo "cloudflared started (token mode)"
  else
    cloudflared tunnel --no-autoupdate --protocol "$CLOUDFLARE_TUNNEL_PROTOCOL" \
      --url "$CLOUDFLARE_TUNNEL_URL" \
      > /var/log/cloudflared.log 2>&1 &
    echo "cloudflared started (trycloudflare mode)"
    url=""
    for _ in {1..15}; do
      url="$(grep -Eo 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' /var/log/cloudflared.log | tail -n 1 || true)"
      if [[ -n "$url" ]]; then
        echo "cloudflared url: $url"
        break
      fi
      sleep 1
    done
    if [[ -z "$url" ]]; then
      echo "cloudflared url not found yet; check /var/log/cloudflared.log"
    fi
  fi
fi
#####
# Solo si quieres que el script arranque ComfyUI:
python3 main.py --listen 0.0.0.0 --port 8188 --use-flash-attention
# pid=$(lsof -t -i :8188) && kill $pid && sleep 1 && kill -9 $pid 2>/dev/null
# python3 main.py --listen 0.0.0.0 --port 8188 --use-flash-attention