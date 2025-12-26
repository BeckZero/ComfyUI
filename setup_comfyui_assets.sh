#!/usr/bin/env bash
set -euo pipefail

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

# fuerza a PyTorch a usar SDPA (scaled dot-product attention) y deja que el backend elija el mejor kernel
export PYTORCH_ENABLE_MPS_FALLBACK=0
export TORCH_LOGS="+attention"        # opcional: logs
export TORCH_SHOW_CPP_STACKTRACES=1   # opcional: debug

python3 -m pip install -r requirements.txt

  
COMFYUI_DIR="${COMFYUI_DIR:-/comfyui}"

if [[ ! -d "$COMFYUI_DIR" ]]; then
  echo "ComfyUI dir not found: $COMFYUI_DIR" >&2
  exit 1
fi

cd "$COMFYUI_DIR"
 
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
 

apt-get update && apt-get install -y lsof
python -m pip install -U GitPython
pip install triton hf 
apt-get update && apt-get install python3.12-dev -y
pip install --no-build-isolation   "git+https://github.com/thu-ml/SageAttention.git@main"
python3 -m pip show SageAttention 



cd /comfyui/custom_nodes/
git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
git clone https://github.com/audioscavenger/save-image-extended-comfyui save-image-extended-comfyui
cd /comfyui/
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


# git fetch upstream
# git checkout master
# git merge upstream/master
# git push origin master



