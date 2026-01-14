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


##descargas de modelos
export HF_HUB_ENABLE_HF_TRANSFER=1
CIVITAI_TOKEN="d860464173933b61d5f3d7111425b1a2"
echo "🚀 Sussy: Lanzando descargas pesadas al fondo. ¡Tú sigue a lo tuyo, cielo!"

# wan_2.1
# # Agrupamos TODO el proceso de descarga en un bloque de segundo plano
# {
#     # Creamos carpetas (dentro del bloque)
#     mkdir -p text_encoders vae diffusion_models loras

#     # Lanzamos las descargas paralelas (como antes)
#     (hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir . && mv split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors /comfyui/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) &

#     (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir . && mv split_files/vae/wan_2.1_vae.safetensors /comfyui/models/vae/wan_2.1_vae.safetensors) &

#     # (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors --local-dir . && mv split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors /comfyui/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors) &

#     # (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors --local-dir . && mv split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors /comfyui/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors) &

#     # (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors --local-dir . && mv split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors /comfyui/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors) &

#     # (hf download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors --local-dir . && mv split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors /comfyui/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors) &

#     (curl -L "https://civitai.com/api/download/models/2376136?token=$CIVITAI_TOKEN" -o /comfyui/models/loras/xxx_amimation_high.safetensors ) &
#     (curl -L "https://civitai.com/api/download/models/2376143?token=$CIVITAI_TOKEN" -o /comfyui/models/loras/xxx_amimation_low.safetensors ) & 

 
#     (curl -L "https://civitai.com/api/download/models/2520801?type=Model&format=GGUF&size=full&fp=fp8&token=$CIVITAI_TOKEN" -o /comfyui/models/diffusion_models/xxx_high.GGUF) &  

#     (curl -L "https://civitai.com/api/download/models/2520805?type=Model&format=GGUF&size=full&fp=fp8&token=$CIVITAI_TOKEN" -o /comfyui/models/diffusion_models/xxx_low.GGUF) & 
 
  
#     # Esperamos a que terminen las descargas dentro de este bloque
#     wait

#     # Limpieza final
#     rm -rf split_files
#     echo -e "\n✅ [Sussy Background]: ¡Modelos Wan 2.2 listos en sus carpetas!"
# } & 





# # z image turbo
# {
#     # Creamos carpetas (dentro del bloque)
#     mkdir -p text_encoders vae diffusion_models loras
#     (hf download Comfy-Org/z_image_turbo split_files/diffusion_models/z_image_turbo_nvfp4.safetensors --local-dir . && mv split_files/diffusion_models/z_image_turbo_nvfp4.safetensors /comfyui/models/diffusion_models/z_image_turbo_nvfp4.safetensors) &

#     # (hf download Comfy-Org/z_image_turbo split_files/loras/z_image_turbo_distill_patch_lora_bf16.safetensors  --local-dir . && mv split_files/loras/z_image_turbo_distill_patch_lora_bf16.safetensors  /comfyui/models/loras/z_image_turbo_distill_patch_lora_bf16.safetensors ) & 

#     # (hf download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors  --local-dir . && mv split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors  /comfyui/models/text_encoders/qwen_3_4b_fp8_mixed.safetensors ) & 

#     # (hf download Comfy-Org/z_image_turbo split_files/vae/ae.safetensors  --local-dir . && mv split_files/vae/ae.safetensors  /comfyui/models/vae/ae.safetensors ) & 

#     (hf download xFanexx/4x-AnimeSharp 4x-AnimeSharp.pth --local-dir . && mv 4x-AnimeSharp.pth /comfyui/models/upscale_models/4x-AnimeSharp.pth) &

  
#     # Esperamos a que terminen las descargas dentro de este bloque
#     wait

#     # Limpieza final
#     rm -rf split_files
#     echo -e "\n✅ [Sussy Background]: ¡Modelos z image turbo 2.2 listos en sus carpetas!"
# } & 




# qwen image 2511
{
    # Creamos carpetas (dentro del bloque)
    mkdir -p text_encoders vae diffusion_models loras 

    (hf download lightx2v/Qwen-Image-Edit-2511-Lightning  qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors --local-dir . && mv qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors) &  
    

    # qwen_2.5_vl_7b_fp8_scaled.safetensors

    # qwen_image_vae.safetensors
    wait

    # Limpieza final
    rm -rf split_files
    echo -e "\n✅ [Sussy Background]: ¡Modelos z image turbo 2.2 listos en sus carpetas!"
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



