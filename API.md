# ComfyUI API (fork BeckZero)

Este documento describe como usar el API HTTP de ComfyUI y los endpoints internos
agregados en este fork para descargar modelos/loras, instalar flash-attn y actualizar
desde upstream.

## Base URL

Asume que ComfyUI corre en:

  http://HOST:PORT

En Vast normalmente es algo como:

  http://IP_PUBLICA:PUERTO

## Autenticacion para endpoints internos

Los endpoints bajo /internal usan el token definido en COMFY_INSTALL_TOKEN.
Puedes enviar el token en:

  Authorization: Bearer <token>

o:

  X-Install-Token: <token>

Si no defines COMFY_INSTALL_TOKEN, los endpoints internos quedan abiertos.

## Generar imagenes (API base)

1) Envia un workflow con POST /prompt
2) Lee el resultado con GET /history/{prompt_id}
3) Descarga la imagen con GET /view

Ejemplo basico:

```bash
curl -X POST http://HOST:PORT/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": {
      "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_turbo_1.0.safetensors"}},
      "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a cinematic photo of a futuristic city"}},
      "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
      "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
      "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": 123456, "steps": 4, "cfg": 1.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
      "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
      "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "turbo"}}
    }
  }'
```

El response contiene `prompt_id`. Usa:

```bash
curl http://HOST:PORT/history/<prompt_id>
```

Para descargar:

```bash
curl "http://HOST:PORT/view?filename=<NOMBRE>.png&subfolder=&type=output" -o out.png
```

## Descargar modelos y LoRAs (endpoint interno)

Habilita:

  COMFY_ALLOW_DOWNLOAD=1

Tambien funciona si COMFY_ALLOW_INSTALL=1.

### Hugging Face

```bash
curl -X POST http://HOST:PORT/internal/download \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "repo_id": "stabilityai/sdxl-turbo",
    "filename": "sd_xl_turbo_1.0.safetensors",
    "folder_type": "checkpoints"
  }'
```

Campos utiles:
  - repo_id (obligatorio)
  - filename (obligatorio)
  - revision (opcional, default: main)
  - token o hf_token (opcional, para repos privados)

### Hugging Face (snapshot / diffusers)

Descarga repos con carpetas (tokenizer/, unet/, etc).

```bash
curl -X POST http://HOST:PORT/internal/download/snapshot \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "stabilityai/sdxl-turbo",
    "folder_type": "diffusers",
    "subfolder": "stabilityai/sdxl-turbo",
    "revision": "main"
  }'
```

Campos utiles:
  - repo_id (obligatorio)
  - revision (opcional, default: main)
  - repo_type (opcional, default: model)
  - folder_type (opcional, default: diffusers)
  - subfolder (opcional, default: repo_id)
  - allow_patterns / ignore_patterns (opcional, lista o string con comas)
  - token o hf_token (opcional, para repos privados)

Nota: requiere `huggingface_hub` instalado en el servidor.

### CivitAI

```bash
curl -X POST http://HOST:PORT/internal/download \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "civitai",
    "civitai_version_id": 123456,
    "folder_type": "loras",
    "civitai_token": "TU_CIVITAI_TOKEN"
  }'
```

Campos utiles:
  - civitai_model_id o civitai_version_id (uno es obligatorio)
  - civitai_filename (opcional, para elegir un archivo exacto)
  - civitai_file_type (opcional, ej: "Model", "LoRA")
  - civitai_token (opcional, para modelos restringidos)

### URL directa

```bash
curl -X POST http://HOST:PORT/internal/download \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "url",
    "url": "https://example.com/model.safetensors",
    "filename": "model.safetensors",
    "folder_type": "checkpoints"
  }'
```

### Estado del download

```bash
curl -H "Authorization: Bearer <token>" \
  http://HOST:PORT/internal/download/<job_id>
```

### Estado del snapshot

```bash
curl -H "Authorization: Bearer <token>" \
  http://HOST:PORT/internal/download/snapshot/<job_id>
```

### folder_type validos

Usa:

```bash
curl http://HOST:PORT/internal/folder_paths
```

Ejemplos comunes: checkpoints, loras, vae, controlnet, embeddings, diffusion_models,
clip_vision, upscale_models, output, input, temp.

## Instalar flash-attn (endpoint interno)

Habilita:

  COMFY_ALLOW_INSTALL=1

Ejemplo:

```bash
curl -X POST http://HOST:PORT/internal/install/flash-attn \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "force": true,
    "cuda_arch_list": "12.0",
    "max_jobs": 4
  }'
```

Estado:

```bash
curl -H "Authorization: Bearer <token>" \
  http://HOST:PORT/internal/install/flash-attn/<job_id>
```

Luego reinicia ComfyUI con:

  python3 main.py --listen 0.0.0.0 --port 8188 --use-flash-attention

## Actualizar desde upstream (endpoint interno)

Habilita:

  COMFY_ALLOW_GIT_UPDATE=1

Configura remotes (recomendado por env):

  COMFY_GIT_UPSTREAM_URL=https://github.com/comfyanonymous/ComfyUI.git
  COMFY_GIT_ORIGIN_URL=https://github.com/BeckZero/ComfyUI.git
  COMFY_GIT_BRANCH=master

Ejemplo (merge + push):

```bash
curl -X POST http://HOST:PORT/internal/git/update \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"merge": true, "push": true}'
```

Estado:

```bash
curl -H "Authorization: Bearer <token>" \
  http://HOST:PORT/internal/git/update/<job_id>
```

Notas:
  - Si hay cambios locales, el job falla.
  - Si hay conflicto, el merge se aborta.
  - Reinicia ComfyUI despues de actualizar.

## Seguridad

Estos endpoints son internos. Exponlos solo si controlas el acceso.
Usa token y firewall / IP allowlist si es posible.
