from aiohttp import web
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Optional
import hashlib
import os
import shutil
import subprocess
import sys
import time
import uuid
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import app.logger
import requests
from api_server.services.terminal_service import TerminalService
from folder_paths import folder_names_and_paths, get_directory_by_type

INSTALL_JOBS: Dict[str, Dict[str, Any]] = {}
INSTALL_LOCK = Lock()
DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
DOWNLOAD_LOCK = Lock()
SNAPSHOT_JOBS: Dict[str, Dict[str, Any]] = {}
SNAPSHOT_LOCK = Lock()
GIT_JOBS: Dict[str, Dict[str, Any]] = {}
GIT_LOCK = Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _normalize_compute_cap(value: str) -> Optional[str]:
    cleaned = value.strip().lower().replace("sm_", "")
    if not cleaned:
        return None
    if "." in cleaned:
        return cleaned
    if cleaned.isdigit():
        if len(cleaned) == 2:
            return f"{cleaned[0]}.{cleaned[1]}"
        if len(cleaned) == 3:
            return f"{cleaned[:2]}.{cleaned[2]}"
    return None


def _format_arch_list(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parts = [p.strip() for p in value.replace(",", ";").split(";") if p.strip()]
    normalized = []
    for part in parts:
        cap = _normalize_compute_cap(part)
        if cap and cap not in normalized:
            normalized.append(cap)
    if not normalized:
        return None
    return ";".join(normalized)


def _detect_cuda_arch_list() -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    normalized = []
    for line in lines:
        cap = _normalize_compute_cap(line)
        if cap and cap not in normalized:
            normalized.append(cap)
    if not normalized:
        return None
    return ";".join(normalized)


def _read_log_tail(path: Path, max_bytes: int = 64 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _run_logged(
    command: list[str],
    log_path: Path,
    env: Dict[str, str],
    cwd: Optional[Path] = None,
    log_command: Optional[list[str]] = None,
) -> int:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(log_command or command)}\n")
        handle.flush()
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    return process.returncode


def _ensure_build_dependencies(log_path: Path, env: Dict[str, str]) -> None:
    if not _env_bool("COMFY_AUTO_INSTALL_DEPS", True):
        return
    if shutil.which("apt-get") is None:
        return
    deps = [
        "build-essential",
        "cmake",
        "ninja-build",
        "git",
        "curl",
        "ca-certificates",
        "python3-dev",
    ]
    env = dict(env)
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    update_code = _run_logged(["apt-get", "update"], log_path, env)
    if update_code != 0:
        raise RuntimeError("apt-get update failed")
    install_code = _run_logged(
        ["apt-get", "install", "-y", "--no-install-recommends", *deps],
        log_path,
        env,
    )
    if install_code != 0:
        raise RuntimeError("apt-get install failed")
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc not found; CUDA toolkit is required to build flash-attn")


def _update_install_job(job_id: str, **updates: Any) -> None:
    with INSTALL_LOCK:
        if job_id in INSTALL_JOBS:
            INSTALL_JOBS[job_id].update(updates)


def _token_ok(request: web.Request) -> bool:
    token = os.getenv("COMFY_INSTALL_TOKEN")
    if not token:
        return True
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        bearer = auth_header.split(" ", 1)[1].strip()
        if bearer == token:
            return True
    header_token = request.headers.get("X-Install-Token", "")
    return header_token == token


def _downloads_allowed() -> bool:
    if os.getenv("COMFY_ALLOW_DOWNLOAD") is not None:
        return _env_bool("COMFY_ALLOW_DOWNLOAD", False)
    return _env_bool("COMFY_ALLOW_INSTALL", False)


def _git_update_allowed() -> bool:
    return _env_bool("COMFY_ALLOW_GIT_UPDATE", False)


def _install_flash_attn_job(
    job_id: str,
    log_path: Path,
    package: str,
    env_overrides: Dict[str, str],
    force: bool,
    extra_args: list[str],
) -> None:
    _update_install_job(job_id, state="running", started_at=_now_iso())
    try:
        job_env = os.environ.copy()
        job_env.update(env_overrides)
        _ensure_build_dependencies(log_path, job_env)

        command = [sys.executable, "-m", "pip", "install", "--no-build-isolation", package]
        if force:
            command.extend(["--force-reinstall", "--no-deps"])
        if extra_args:
            command.extend(extra_args)

        return_code = _run_logged(command, log_path, job_env)
        if return_code != 0:
            raise RuntimeError(f"pip install failed with code {return_code}")

        _update_install_job(
            job_id,
            state="succeeded",
            finished_at=_now_iso(),
            return_code=return_code,
        )
    except Exception as exc:
        _update_install_job(
            job_id,
            state="failed",
            finished_at=_now_iso(),
            error=str(exc),
        )


def _sanitize_subfolder(subfolder: Optional[str]) -> str:
    if not subfolder:
        return ""
    cleaned = subfolder.strip().strip("/").strip("\\")
    if not cleaned:
        return ""
    if os.path.isabs(cleaned):
        raise ValueError("subfolder must be relative")
    parts = Path(cleaned).parts
    if any(part == ".." for part in parts):
        raise ValueError("subfolder must not contain '..'")
    return os.path.join(*parts)


def _sanitize_filename(name: str) -> str:
    cleaned = os.path.basename(name.strip())
    if not cleaned:
        return "download.bin"
    return cleaned


def _resolve_download_dir(folder_type: str, subfolder: Optional[str]) -> Path:
    if folder_type in ("output", "input", "temp"):
        base_path = Path(get_directory_by_type(folder_type))
    else:
        entry = folder_names_and_paths.get(folder_type)
        if not entry:
            raise ValueError(f"Invalid folder_type: {folder_type}")
        base_path = Path(entry[0][0])
    safe_subfolder = _sanitize_subfolder(subfolder)
    target = base_path / safe_subfolder if safe_subfolder else base_path
    target.mkdir(parents=True, exist_ok=True)
    return target


def _strip_token_from_url(url: str) -> str:
    parsed = urlparse(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"token", "access_token"}
    ]
    return urlunparse(parsed._replace(query=urlencode(query)))


def _redact_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    safe_url = _strip_token_from_url(url)
    parsed = urlparse(safe_url)
    if parsed.scheme and parsed.netloc and "@" in parsed.netloc:
        _, host = parsed.netloc.rsplit("@", 1)
        return urlunparse(parsed._replace(netloc=f"***@{host}"))
    return safe_url


def _quote_path(value: str) -> str:
    return "/".join(quote(part) for part in value.split("/"))


def _build_hf_url(repo_id: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{_quote_path(repo_id)}/resolve/{quote(revision, safe='')}/{_quote_path(filename)}"


def _fetch_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _pick_civitai_version(versions: list[dict[str, Any]], version_id: Optional[int]) -> dict[str, Any]:
    if version_id is not None:
        for version in versions:
            if str(version.get("id")) == str(version_id):
                return version
        raise ValueError(f"CivitAI version_id not found: {version_id}")
    def _sort_key(version: dict[str, Any]) -> str:
        return version.get("publishedAt") or version.get("updatedAt") or version.get("createdAt") or ""
    return sorted(versions, key=_sort_key, reverse=True)[0]


def _pick_civitai_file(
    files: list[dict[str, Any]],
    preferred_filename: Optional[str],
    preferred_type: Optional[str],
) -> dict[str, Any]:
    if preferred_filename:
        for file_info in files:
            if file_info.get("name") == preferred_filename:
                return file_info
    if preferred_type:
        for file_info in files:
            if str(file_info.get("type", "")).lower() == preferred_type.lower():
                return file_info
    return files[0]


def _resolve_civitai_download(
    model_id: Optional[int],
    version_id: Optional[int],
    token: Optional[str],
    preferred_filename: Optional[str],
    preferred_type: Optional[str],
) -> tuple[str, str]:
    headers = {"User-Agent": "ComfyUI-Internal-Downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if version_id is not None:
        data = _fetch_json(f"https://civitai.com/api/v1/model-versions/{version_id}", headers)
    else:
        if model_id is None:
            raise ValueError("civitai_model_id or civitai_version_id is required")
        data = _fetch_json(f"https://civitai.com/api/v1/models/{model_id}", headers)
        versions = data.get("modelVersions") or []
        if not versions:
            raise ValueError(f"No model versions found for civitai_model_id {model_id}")
        data = _pick_civitai_version(versions, version_id)

    files = data.get("files") or []
    if not files:
        raise ValueError("No files found for the requested CivitAI model/version")
    selected = _pick_civitai_file(files, preferred_filename, preferred_type)
    download_url = selected.get("downloadUrl") or selected.get("download_url")
    if not download_url:
        raise ValueError("CivitAI download URL not found")

    if token and "token=" not in download_url:
        parsed = urlparse(download_url)
        query = parse_qsl(parsed.query, keep_blank_values=True)
        query.append(("token", token))
        download_url = urlunparse(parsed._replace(query=urlencode(query)))

    filename = _sanitize_filename(preferred_filename or selected.get("name") or os.path.basename(urlparse(download_url).path))
    return download_url, filename


def _update_download_job(job_id: str, **updates: Any) -> None:
    with DOWNLOAD_LOCK:
        if job_id in DOWNLOAD_JOBS:
            DOWNLOAD_JOBS[job_id].update(updates)


def _update_snapshot_job(job_id: str, **updates: Any) -> None:
    with SNAPSHOT_LOCK:
        if job_id in SNAPSHOT_JOBS:
            SNAPSHOT_JOBS[job_id].update(updates)


def _parse_patterns(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = value.replace("\n", ",").split(",")
        return [part.strip() for part in parts if part.strip()]
    return []


def _count_files_and_bytes(path: Path) -> tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_files += 1
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue
    return total_files, total_bytes


def _snapshot_download_job(
    job_id: str,
    log_path: Path,
    repo_id: str,
    revision: str,
    repo_type: str,
    token: Optional[str],
    allow_patterns: list[str],
    ignore_patterns: list[str],
    folder_type: str,
    subfolder: Optional[str],
) -> None:
    _update_snapshot_job(job_id, state="running", started_at=_now_iso())
    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc

        target_subfolder = subfolder if subfolder else repo_id
        dest_dir = _resolve_download_dir(folder_type, target_subfolder)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"repo_id: {repo_id}\n")
            handle.write(f"revision: {revision}\n")
            handle.write(f"repo_type: {repo_type}\n")
            handle.write(f"dest: {dest_dir}\n")
            if allow_patterns:
                handle.write(f"allow_patterns: {allow_patterns}\n")
            if ignore_patterns:
                handle.write(f"ignore_patterns: {ignore_patterns}\n")

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=allow_patterns or None,
            ignore_patterns=ignore_patterns or None,
        )

        files_count, total_bytes = _count_files_and_bytes(dest_dir)
        _update_snapshot_job(
            job_id,
            state="succeeded",
            finished_at=_now_iso(),
            files_count=files_count,
            total_bytes=total_bytes,
            path=str(dest_dir),
        )
    except Exception as exc:
        _update_snapshot_job(
            job_id,
            state="failed",
            finished_at=_now_iso(),
            error=str(exc),
        )


def _is_safe_git_ref(value: str) -> bool:
    if not value:
        return False
    if value.startswith("/") or value.endswith("/") or value.endswith(".lock"):
        return False
    if ".." in value:
        return False
    if any(char in value for char in (" ", "~", "^", ":", "?", "*", "\\", "[", "]", "@")):
        return False
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_git_remote(
    repo_dir: Path,
    remote: str,
    url: Optional[str],
    log_path: Path,
    env: Dict[str, str],
) -> None:
    safe_url = _redact_url(url)
    result = subprocess.run(
        ["git", "remote", "get-url", remote],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
    )
    if result.returncode != 0:
        if not url:
            raise RuntimeError(f"Remote '{remote}' is not configured")
        code = _run_logged(
            ["git", "remote", "add", remote, url],
            log_path,
            env,
            cwd=repo_dir,
            log_command=["git", "remote", "add", remote, safe_url or ""],
        )
        if code != 0:
            raise RuntimeError(f"Failed to add remote '{remote}'")
    elif url:
        code = _run_logged(
            ["git", "remote", "set-url", remote, url],
            log_path,
            env,
            cwd=repo_dir,
            log_command=["git", "remote", "set-url", remote, safe_url or ""],
        )
        if code != 0:
            raise RuntimeError(f"Failed to set remote '{remote}' URL")


def _update_git_job(job_id: str, **updates: Any) -> None:
    with GIT_LOCK:
        if job_id in GIT_JOBS:
            GIT_JOBS[job_id].update(updates)


def _git_update_job(
    job_id: str,
    log_path: Path,
    branch: str,
    merge: bool,
    push: bool,
    upstream_remote: str,
    origin_remote: str,
    upstream_url: Optional[str],
    origin_url: Optional[str],
) -> None:
    _update_git_job(job_id, state="running", started_at=_now_iso())
    env = os.environ.copy()
    repo_dir = _repo_root()
    try:
        if shutil.which("git") is None:
            raise RuntimeError("git not found in PATH")
        if not (repo_dir / ".git").exists():
            raise RuntimeError(f"Repository not found at {repo_dir}")

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
        )
        if status.returncode != 0:
            raise RuntimeError("git status failed")
        if status.stdout.strip():
            raise RuntimeError("Working tree has local changes; aborting update")

        _ensure_git_remote(repo_dir, upstream_remote, upstream_url, log_path, env)
        if push or origin_url:
            _ensure_git_remote(repo_dir, origin_remote, origin_url, log_path, env)

        current_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
        )
        if current_branch.returncode != 0:
            raise RuntimeError("Failed to determine current branch")
        if current_branch.stdout.strip() != branch:
            raise RuntimeError(f"Current branch is '{current_branch.stdout.strip()}', expected '{branch}'")

        fetch_code = _run_logged(["git", "fetch", upstream_remote], log_path, env, cwd=repo_dir)
        if fetch_code != 0:
            raise RuntimeError("git fetch failed")

        merge_target = f"{upstream_remote}/{branch}"
        merge_command = ["git", "merge"]
        if merge:
            merge_command.append("--no-edit")
        else:
            merge_command.append("--ff-only")
        merge_command.append(merge_target)

        merge_code = _run_logged(merge_command, log_path, env, cwd=repo_dir)
        if merge_code != 0:
            if merge and (repo_dir / ".git" / "MERGE_HEAD").exists():
                _run_logged(["git", "merge", "--abort"], log_path, env, cwd=repo_dir)
            raise RuntimeError("git merge failed")

        if push:
            push_code = _run_logged(["git", "push", origin_remote, branch], log_path, env, cwd=repo_dir)
            if push_code != 0:
                raise RuntimeError("git push failed")

        _update_git_job(
            job_id,
            state="succeeded",
            finished_at=_now_iso(),
            return_code=0,
        )
    except Exception as exc:
        _update_git_job(
            job_id,
            state="failed",
            finished_at=_now_iso(),
            error=str(exc),
        )


def _download_file_job(
    job_id: str,
    url: str,
    dest_dir: Path,
    filename: str,
    headers: Dict[str, str],
    expected_sha256: Optional[str],
    overwrite: bool,
) -> None:
    dest_path = dest_dir / filename
    temp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    _update_download_job(job_id, state="running", started_at=_now_iso())
    try:
        if dest_path.exists() and not overwrite:
            _update_download_job(
                job_id,
                state="succeeded",
                finished_at=_now_iso(),
                skipped=True,
                bytes_downloaded=dest_path.stat().st_size,
                total_bytes=dest_path.stat().st_size,
                path=str(dest_path),
            )
            return

        hasher = hashlib.sha256() if expected_sha256 else None
        bytes_downloaded = 0
        with requests.get(url, headers=headers, stream=True, timeout=(10, 600)) as response:
            response.raise_for_status()
            total_header = response.headers.get("content-length")
            total_bytes = int(total_header) if total_header and total_header.isdigit() else None
            _update_download_job(job_id, total_bytes=total_bytes)
            last_update = time.monotonic()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_downloaded += len(chunk)
                    if hasher:
                        hasher.update(chunk)
                    now = time.monotonic()
                    if now - last_update >= 0.5:
                        _update_download_job(job_id, bytes_downloaded=bytes_downloaded)
                        last_update = now

        if hasher:
            digest = hasher.hexdigest()
            if expected_sha256 and digest.lower() != expected_sha256.lower():
                temp_path.unlink(missing_ok=True)
                raise RuntimeError("sha256 mismatch")
        else:
            digest = None

        temp_path.replace(dest_path)
        _update_download_job(
            job_id,
            state="succeeded",
            finished_at=_now_iso(),
            bytes_downloaded=bytes_downloaded,
            total_bytes=bytes_downloaded,
            sha256=digest,
            path=str(dest_path),
        )
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        _update_download_job(
            job_id,
            state="failed",
            finished_at=_now_iso(),
            error=str(exc),
        )


class InternalRoutes:
    '''
    The top level web router for internal routes: /internal/*
    The endpoints here should NOT be depended upon. It is for ComfyUI frontend use only.
    Check README.md for more information.
    '''

    def __init__(self, prompt_server):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: Optional[web.Application] = None
        self.prompt_server = prompt_server
        self.terminal_service = TerminalService(prompt_server)

    def setup_routes(self):
        @self.routes.get('/logs')
        async def get_logs(request):
            return web.json_response("".join([(l["t"] + " - " + l["m"]) for l in app.logger.get_logs()]))

        @self.routes.get('/logs/raw')
        async def get_raw_logs(request):
            self.terminal_service.update_size()
            return web.json_response({
                "entries": list(app.logger.get_logs()),
                "size": {"cols": self.terminal_service.cols, "rows": self.terminal_service.rows}
            })

        @self.routes.patch('/logs/subscribe')
        async def subscribe_logs(request):
            json_data = await request.json()
            client_id = json_data["clientId"]
            enabled = json_data["enabled"]
            if enabled:
                self.terminal_service.subscribe(client_id)
            else:
                self.terminal_service.unsubscribe(client_id)

            return web.Response(status=200)


        @self.routes.get('/folder_paths')
        async def get_folder_paths(request):
            response = {}
            for key in folder_names_and_paths:
                response[key] = folder_names_and_paths[key][0]
            return web.json_response(response)

        @self.routes.get('/files/{directory_type}')
        async def get_files(request: web.Request) -> web.Response:
            directory_type = request.match_info['directory_type']
            if directory_type not in ("output", "input", "temp"):
                return web.json_response({"error": "Invalid directory type"}, status=400)

            directory = get_directory_by_type(directory_type)

            def is_visible_file(entry: os.DirEntry) -> bool:
                """Filter out hidden files (e.g., .DS_Store on macOS)."""
                return entry.is_file() and not entry.name.startswith('.')

            sorted_files = sorted(
                (entry for entry in os.scandir(directory) if is_visible_file(entry)),
                key=lambda entry: -entry.stat().st_mtime
            )
            return web.json_response([entry.name for entry in sorted_files], status=200)

        @self.routes.post('/install/flash-attn')
        async def install_flash_attn(request: web.Request) -> web.Response:
            if not _env_bool("COMFY_ALLOW_INSTALL", False):
                return web.json_response({"error": "Install API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            payload: Dict[str, Any] = {}
            if request.can_read_body:
                try:
                    payload = await request.json()
                except Exception:
                    return web.json_response({"error": "Invalid JSON body"}, status=400)

            force = _parse_bool(payload.get("force"), False)
            package = payload.get("package", "flash-attn")
            if not isinstance(package, str) or not package.strip():
                return web.json_response({"error": "Invalid package"}, status=400)

            extra_args = payload.get("extra_args", [])
            if not isinstance(extra_args, list) or any(not isinstance(arg, str) for arg in extra_args):
                return web.json_response({"error": "extra_args must be a list of strings"}, status=400)

            max_jobs = payload.get("max_jobs")
            if max_jobs is not None:
                try:
                    max_jobs = int(max_jobs)
                except (TypeError, ValueError):
                    return web.json_response({"error": "max_jobs must be an integer"}, status=400)
                if max_jobs <= 0:
                    return web.json_response({"error": "max_jobs must be > 0"}, status=400)

            requested_arch = payload.get("cuda_arch_list") or payload.get("cuda_architectures")
            cuda_arch_list = _format_arch_list(requested_arch)
            if cuda_arch_list is None:
                cuda_arch_list = _format_arch_list(
                    os.getenv("COMFY_TORCH_CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST")
                )
            if cuda_arch_list is None:
                cuda_arch_list = _detect_cuda_arch_list()

            env_overrides: Dict[str, str] = {}
            if cuda_arch_list:
                env_overrides["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
            if max_jobs:
                env_overrides["MAX_JOBS"] = str(max_jobs)

            with INSTALL_LOCK:
                running_job = next(
                    (job for job in INSTALL_JOBS.values() if job.get("state") == "running"),
                    None,
                )
                if running_job:
                    return web.json_response(
                        {"error": "Install already running", "job_id": running_job.get("id")},
                        status=409,
                    )

                job_id = uuid.uuid4().hex
                log_dir = Path(get_directory_by_type("output")) / "install"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"flash-attn-{job_id}.log"

                INSTALL_JOBS[job_id] = {
                    "id": job_id,
                    "state": "queued",
                    "created_at": _now_iso(),
                    "started_at": None,
                    "finished_at": None,
                    "return_code": None,
                    "error": None,
                    "package": package,
                    "force": force,
                    "cuda_arch_list": cuda_arch_list,
                    "log_path": str(log_path),
                }

            thread = Thread(
                target=_install_flash_attn_job,
                args=(job_id, log_path, package, env_overrides, force, extra_args),
                daemon=True,
            )
            thread.start()

            return web.json_response(
                {
                    "job_id": job_id,
                    "status_url": f"/internal/install/flash-attn/{job_id}",
                    "cuda_arch_list": cuda_arch_list,
                },
                status=202,
            )

        @self.routes.get('/install/flash-attn/{job_id}')
        async def get_install_flash_attn(request: web.Request) -> web.Response:
            if not _env_bool("COMFY_ALLOW_INSTALL", False):
                return web.json_response({"error": "Install API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            job_id = request.match_info["job_id"]
            with INSTALL_LOCK:
                job = INSTALL_JOBS.get(job_id)
                if job is None:
                    return web.json_response({"error": "Job not found"}, status=404)
                job_data = dict(job)

            log_path = Path(job_data.get("log_path", ""))
            job_data["log_tail"] = _read_log_tail(log_path)
            return web.json_response(job_data, status=200)

        @self.routes.post('/download')
        async def download_model(request: web.Request) -> web.Response:
            if not _downloads_allowed():
                return web.json_response({"error": "Download API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            try:
                payload = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)

            if not isinstance(payload, dict):
                return web.json_response({"error": "Invalid request body"}, status=400)

            folder_type = payload.get("folder_type", "checkpoints")
            if not isinstance(folder_type, str) or not folder_type.strip():
                return web.json_response({"error": "folder_type must be a string"}, status=400)

            allowed_folders = set(folder_names_and_paths.keys()) | {"output", "input", "temp"}
            if folder_type not in allowed_folders:
                return web.json_response({"error": f"Invalid folder_type: {folder_type}"}, status=400)

            subfolder = payload.get("subfolder")
            overwrite = _parse_bool(payload.get("overwrite"), False)
            expected_sha256 = payload.get("sha256")
            if expected_sha256 is not None and not isinstance(expected_sha256, str):
                return web.json_response({"error": "sha256 must be a string"}, status=400)

            source = payload.get("source")
            if not source:
                if payload.get("civitai_version_id") or payload.get("civitai_model_id"):
                    source = "civitai"
                elif payload.get("repo_id") or payload.get("hf_repo"):
                    source = "huggingface"
                elif payload.get("url"):
                    source = "url"
            if not source:
                return web.json_response({"error": "source is required"}, status=400)
            source = str(source).lower()

            url = None
            filename = None
            headers = {"User-Agent": "ComfyUI-Internal-Downloader"}

            if source == "url":
                url = payload.get("url")
                if not isinstance(url, str) or not url.strip():
                    return web.json_response({"error": "url is required"}, status=400)
                filename = payload.get("filename") or payload.get("dest_filename")
                if not filename:
                    path_name = os.path.basename(urlparse(url).path)
                    filename = path_name if path_name else "download.bin"
            elif source == "huggingface":
                repo_id = payload.get("repo_id") or payload.get("hf_repo")
                hf_filename = payload.get("filename")
                revision = payload.get("revision", "main")
                if not isinstance(repo_id, str) or not repo_id.strip():
                    return web.json_response({"error": "repo_id is required"}, status=400)
                if not isinstance(hf_filename, str) or not hf_filename.strip():
                    return web.json_response({"error": "filename is required"}, status=400)
                if not isinstance(revision, str) or not revision.strip():
                    return web.json_response({"error": "revision must be a string"}, status=400)
                token = payload.get("token") or payload.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                url = _build_hf_url(repo_id.strip(), hf_filename.strip(), revision.strip())
                filename = payload.get("dest_filename") or hf_filename
            elif source == "civitai":
                model_id = payload.get("civitai_model_id")
                version_id = payload.get("civitai_version_id")
                civitai_filename = payload.get("civitai_filename")
                civitai_file_type = payload.get("civitai_file_type")
                if model_id is None and version_id is None:
                    return web.json_response({"error": "civitai_model_id or civitai_version_id is required"}, status=400)
                try:
                    model_id_int = int(model_id) if model_id is not None else None
                except (TypeError, ValueError):
                    return web.json_response({"error": "civitai_model_id must be an integer"}, status=400)
                try:
                    version_id_int = int(version_id) if version_id is not None else None
                except (TypeError, ValueError):
                    return web.json_response({"error": "civitai_version_id must be an integer"}, status=400)
                token = payload.get("token") or payload.get("civitai_token") or os.getenv("CIVITAI_TOKEN")
                try:
                    url, filename = _resolve_civitai_download(
                        model_id_int,
                        version_id_int,
                        token,
                        civitai_filename,
                        civitai_file_type,
                    )
                except Exception as exc:
                    return web.json_response({"error": str(exc)}, status=400)
                filename = payload.get("dest_filename") or filename
            else:
                return web.json_response({"error": f"Unsupported source: {source}"}, status=400)

            filename = _sanitize_filename(str(filename))
            try:
                dest_dir = _resolve_download_dir(folder_type, subfolder)
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=400)

            job_id = uuid.uuid4().hex
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOBS[job_id] = {
                    "id": job_id,
                    "state": "queued",
                    "created_at": _now_iso(),
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "source": source,
                    "url": _strip_token_from_url(url),
                    "folder_type": folder_type,
                    "subfolder": subfolder,
                    "filename": filename,
                    "bytes_downloaded": 0,
                    "total_bytes": None,
                    "sha256": None,
                    "path": None,
                    "skipped": False,
                }

            thread = Thread(
                target=_download_file_job,
                args=(job_id, url, dest_dir, filename, headers, expected_sha256, overwrite),
                daemon=True,
            )
            thread.start()

            return web.json_response(
                {
                    "job_id": job_id,
                    "status_url": f"/internal/download/{job_id}",
                    "folder_type": folder_type,
                    "subfolder": subfolder,
                    "filename": filename,
                },
                status=202,
            )

        @self.routes.get('/download/{job_id}')
        async def get_download_status(request: web.Request) -> web.Response:
            if not _downloads_allowed():
                return web.json_response({"error": "Download API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            job_id = request.match_info["job_id"]
            with DOWNLOAD_LOCK:
                job = DOWNLOAD_JOBS.get(job_id)
                if job is None:
                    return web.json_response({"error": "Job not found"}, status=404)
                job_data = dict(job)

            return web.json_response(job_data, status=200)

        @self.routes.post('/download/snapshot')
        async def download_snapshot(request: web.Request) -> web.Response:
            if not _downloads_allowed():
                return web.json_response({"error": "Download API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            try:
                payload = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)

            if not isinstance(payload, dict):
                return web.json_response({"error": "Invalid request body"}, status=400)

            repo_id = payload.get("repo_id") or payload.get("hf_repo")
            if not isinstance(repo_id, str) or not repo_id.strip():
                return web.json_response({"error": "repo_id is required"}, status=400)

            revision = payload.get("revision", "main")
            if not isinstance(revision, str) or not revision.strip():
                return web.json_response({"error": "revision must be a string"}, status=400)

            repo_type = payload.get("repo_type", "model")
            if not isinstance(repo_type, str) or not repo_type.strip():
                return web.json_response({"error": "repo_type must be a string"}, status=400)

            folder_type = payload.get("folder_type", "diffusers")
            if not isinstance(folder_type, str) or not folder_type.strip():
                return web.json_response({"error": "folder_type must be a string"}, status=400)

            allowed_folders = set(folder_names_and_paths.keys()) | {"output", "input", "temp"}
            if folder_type not in allowed_folders:
                return web.json_response({"error": f"Invalid folder_type: {folder_type}"}, status=400)

            subfolder = payload.get("subfolder")
            if subfolder:
                try:
                    _sanitize_subfolder(str(subfolder))
                except ValueError as exc:
                    return web.json_response({"error": str(exc)}, status=400)

            token = payload.get("token") or payload.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            allow_patterns = _parse_patterns(payload.get("allow_patterns"))
            ignore_patterns = _parse_patterns(payload.get("ignore_patterns"))

            job_id = uuid.uuid4().hex
            log_dir = Path(get_directory_by_type("output")) / "snapshots"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"snapshot-{job_id}.log"

            with SNAPSHOT_LOCK:
                SNAPSHOT_JOBS[job_id] = {
                    "id": job_id,
                    "state": "queued",
                    "created_at": _now_iso(),
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "repo_id": repo_id,
                    "revision": revision,
                    "repo_type": repo_type,
                    "folder_type": folder_type,
                    "subfolder": subfolder,
                    "allow_patterns": allow_patterns,
                    "ignore_patterns": ignore_patterns,
                    "files_count": None,
                    "total_bytes": None,
                    "path": None,
                    "log_path": str(log_path),
                }

            thread = Thread(
                target=_snapshot_download_job,
                args=(
                    job_id,
                    log_path,
                    repo_id.strip(),
                    revision.strip(),
                    repo_type.strip(),
                    token,
                    allow_patterns,
                    ignore_patterns,
                    folder_type,
                    subfolder,
                ),
                daemon=True,
            )
            thread.start()

            return web.json_response(
                {
                    "job_id": job_id,
                    "status_url": f"/internal/download/snapshot/{job_id}",
                    "folder_type": folder_type,
                    "subfolder": subfolder,
                },
                status=202,
            )

        @self.routes.get('/download/snapshot/{job_id}')
        async def get_download_snapshot_status(request: web.Request) -> web.Response:
            if not _downloads_allowed():
                return web.json_response({"error": "Download API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            job_id = request.match_info["job_id"]
            with SNAPSHOT_LOCK:
                job = SNAPSHOT_JOBS.get(job_id)
                if job is None:
                    return web.json_response({"error": "Job not found"}, status=404)
                job_data = dict(job)

            log_path = Path(job_data.get("log_path", ""))
            job_data["log_tail"] = _read_log_tail(log_path)
            return web.json_response(job_data, status=200)

        @self.routes.post('/git/update')
        async def git_update(request: web.Request) -> web.Response:
            if not _git_update_allowed():
                return web.json_response({"error": "Git update API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            payload: Dict[str, Any] = {}
            if request.can_read_body:
                try:
                    payload = await request.json()
                except Exception:
                    return web.json_response({"error": "Invalid JSON body"}, status=400)

            branch = payload.get("branch") or os.getenv("COMFY_GIT_BRANCH") or "master"
            if not isinstance(branch, str) or not _is_safe_git_ref(branch):
                return web.json_response({"error": "Invalid branch name"}, status=400)

            merge = _parse_bool(payload.get("merge"), False)
            push = _parse_bool(payload.get("push"), False)

            upstream_remote = payload.get("upstream_remote") or os.getenv("COMFY_GIT_UPSTREAM_REMOTE") or "upstream"
            origin_remote = payload.get("origin_remote") or os.getenv("COMFY_GIT_ORIGIN_REMOTE") or "origin"
            if not isinstance(upstream_remote, str) or not upstream_remote.strip():
                return web.json_response({"error": "Invalid upstream_remote"}, status=400)
            if not isinstance(origin_remote, str) or not origin_remote.strip():
                return web.json_response({"error": "Invalid origin_remote"}, status=400)

            allow_remote_update = _env_bool("COMFY_ALLOW_GIT_REMOTE_UPDATE", False)
            upstream_url = None
            origin_url = None
            if allow_remote_update:
                upstream_url = payload.get("upstream_url") or os.getenv("COMFY_GIT_UPSTREAM_URL")
                origin_url = payload.get("origin_url") or os.getenv("COMFY_GIT_ORIGIN_URL")
            else:
                upstream_url = os.getenv("COMFY_GIT_UPSTREAM_URL")
                origin_url = os.getenv("COMFY_GIT_ORIGIN_URL")

            job_id = uuid.uuid4().hex
            log_dir = Path(get_directory_by_type("output")) / "git"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"git-update-{job_id}.log"

            with GIT_LOCK:
                GIT_JOBS[job_id] = {
                    "id": job_id,
                    "state": "queued",
                    "created_at": _now_iso(),
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "branch": branch,
                    "merge": merge,
                    "push": push,
                    "upstream_remote": upstream_remote,
                    "origin_remote": origin_remote,
                    "log_path": str(log_path),
                }

            thread = Thread(
                target=_git_update_job,
                args=(
                    job_id,
                    log_path,
                    branch,
                    merge,
                    push,
                    upstream_remote,
                    origin_remote,
                    upstream_url,
                    origin_url,
                ),
                daemon=True,
            )
            thread.start()

            return web.json_response(
                {
                    "job_id": job_id,
                    "status_url": f"/internal/git/update/{job_id}",
                    "branch": branch,
                    "merge": merge,
                    "push": push,
                },
                status=202,
            )

        @self.routes.get('/git/update/{job_id}')
        async def get_git_update_status(request: web.Request) -> web.Response:
            if not _git_update_allowed():
                return web.json_response({"error": "Git update API disabled"}, status=403)
            if not _token_ok(request):
                return web.json_response({"error": "Invalid token"}, status=403)

            job_id = request.match_info["job_id"]
            with GIT_LOCK:
                job = GIT_JOBS.get(job_id)
                if job is None:
                    return web.json_response({"error": "Job not found"}, status=404)
                job_data = dict(job)

            log_path = Path(job_data.get("log_path", ""))
            job_data["log_tail"] = _read_log_tail(log_path)
            return web.json_response(job_data, status=200)

    def get_app(self):
        if self._app is None:
            self._app = web.Application()
            self.setup_routes()
            self._app.add_routes(self.routes)
        return self._app
