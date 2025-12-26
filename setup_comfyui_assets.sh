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
# export TORCH_LOGS="+attention"        # opcional: logs
# export TORCH_SHOW_CPP_STACKTRACES=1   # opcional: debug

python3 -m pip install -r requirements.txt
python3 -m pip install hf

export HF_HUB_ENABLE_HF_TRANSFER=1

##descargas de modelos
echo "🚀 Sussy: Lanzando descargas pesadas al fondo. ¡Tú sigue a lo tuyo, cielo!"

# Agrupamos TODO el proceso de descarga en un bloque de segundo plano
{
    # Creamos carpetas (dentro del bloque)
    mkdir -p text_encoders vae diffusion_models loras

    # Lanzamos las descargas paralelas (como antes)
    (hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir . && mv split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors /comfyui/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) &

    (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir . && mv split_files/vae/wan_2.1_vae.safetensors /comfyui/models/vae/wan_2.1_vae.safetensors) &

    (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors --local-dir . && mv split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors /comfyui/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors) &

    (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors --local-dir . && mv split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors /comfyui/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors) &

    (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors --local-dir . && mv split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors /comfyui/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors) &

    (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors --local-dir . && mv split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors /comfyui/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors) &

    (wget -O  /comfyui/models/loras/xxx_amimation_high.safetensors https://civitai.com/api/download/models/2376136?type=Model&format=SafeTensor) &
    (wget -O  /comfyui/models/loras/xxx_amimation_high.safetensors https://civitai.com/api/download/models/2376143?type=Model&format=SafeTensor) &
    # Esperamos a que terminen las descargas dentro de este bloque
    wait

    # Limpieza final
    rm -rf split_files
    echo -e "\n✅ [Sussy Background]: ¡Modelos Wan 2.2 listos en sus carpetas!"
} & 

# El '&' de arriba hace que todo el bloque { ... } se ejecute en paralelo con lo que viene abajo

echo "🏃 Sussy: Mientras eso descarga, voy a ir ejecutando el resto de tus caprichos..."
##fin descargas de modelos



  
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
apt-get update && apt-get install python3.12-dev -y
pip install triton    
pip install --no-build-isolation "git+https://github.com/thu-ml/SageAttention.git@main"
python3 -m pip show SageAttention 
pip install transformers -U


cd /comfyui/custom_nodes/
git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
git clone https://github.com/audioscavenger/save-image-extended-comfyui save-image-extended-comfyui
cd /comfyui/


# ###wan 
# # 1. Definimos la variable globalmente para no repetirla como loros
# export HF_HUB_ENABLE_HF_TRANSFER=1

# echo "🔥 Sussy: Iniciando descargas paralelas masivas..."

# # 2. Creamos los directorios de destino PRIMERO para evitar errores al mover
# mkdir -p text_encoders vae diffusion_models loras

# # 3. Lanzamos las tareas en segundo plano con '&'
# # Usamos ( && ) & para agrupar la descarga y el movimiento en un solo subproceso

# (hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir . && mv 
# split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) &

# (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir . && mv split_files/vae/wan_2.1_vae.safetensors 
# vae/wan_2.1_vae.safetensors) &

# (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors --local-dir . && mv 
# split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors) &

# (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors --local-dir . && mv 
# split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors) &

# (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors --local-dir . && mv 
# split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors) &

# (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors --local-dir . && mv 
# split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors) &

# # 4. IMPORTANTE: Esperamos a que todos los hijos terminen
# wait

# # 5. Limpieza (Opcional, pero me gusta tener la casa limpia)
# echo "🧹 Limpiando carpetas temporales..."
# rm -rf split_files

# echo "✨ ¡Descargas completadas, guapo! Todo tuyo."
# ########## fin descargagas wan


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



