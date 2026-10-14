from __future__ import annotations

import asyncio
import ipaddress
import io
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    get_cached_runtime,
)


LOGGER = logging.getLogger("mirai-irodori-server")

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".webm",
}
LATENT_EXTENSIONS = {".pt", ".pth"}
SPEAKER_EMBEDDING_SUFFIX = ".speaker.safetensors"
# Accept the common singular typo while keeping the runtime-facing path canonical.
SPEAKER_EMBEDDING_TYPO_SUFFIX = ".speaker.safestensor"
NO_REF_IDS = {"", "none", "no_ref", "no-ref", "null", "text-only"}
DEFAULT_HF_CHECKPOINT = "Aratako/Irodori-TTS-v4-Small"
MAX_VOICE_DESIGN_CANDIDATES = 32


def _discover_local_ipv4_addresses() -> tuple[str, ...]:
    """Return usable local IPv4 addresses, with the default-route address first."""
    candidates: list[str] = []

    # A UDP connect selects the OS default route without sending any packets.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            candidates.append(str(probe.getsockname()[0]))
    except OSError:
        pass

    try:
        candidates.extend(
            str(info[4][0])
            for info in socket.getaddrinfo(
                socket.gethostname(),
                None,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        )
    except OSError:
        pass

    addresses: list[str] = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if (
            not isinstance(address, ipaddress.IPv4Address)
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
        ):
            continue
        normalized = str(address)
        if normalized not in addresses:
            addresses.append(normalized)
    return tuple(addresses)


def _connection_urls(bind_host: str, port: int) -> tuple[str, ...]:
    host = bind_host.strip() or "0.0.0.0"
    if host in {"0.0.0.0", "::", "[::]"}:
        hosts = _discover_local_ipv4_addresses()
    elif host.casefold() in {"localhost", "127.0.0.1", "::1", "[::1]"}:
        hosts = ()
    else:
        hosts = (host,)

    urls: list[str] = []
    for item in hosts:
        url_host = item if ":" not in item or item.startswith("[") else f"[{item}]"
        url = f"http://{url_host}:{port}"
        if url not in urls:
            urls.append(url)
    return tuple(urls)


def _print_connection_urls(bind_host: str, port: int) -> None:
    print("MiRAI Head connection URL(s):", flush=True)
    urls = _connection_urls(bind_host, port)
    if urls:
        for url in urls:
            print(f"  {url}", flush=True)
    else:
        print("  No LAN IPv4 address detected.", flush=True)
    print(flush=True)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(value: str | None, label: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must be a number or blank") from exc


def _optional_int(value: str | None, label: str) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must be an integer or blank") from exc


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


@dataclass(frozen=True)
class VoiceSource:
    voice_id: str
    mode: Literal["speaker_embedding", "latents", "audio", "no_ref"]
    paths: tuple[Path, ...] = ()

    def sampling_kwargs(self) -> dict[str, object]:
        string_paths = [str(path) for path in self.paths]
        if self.mode == "speaker_embedding":
            return {"ref_embed": string_paths[0]}
        if self.mode == "latents":
            if len(string_paths) == 1:
                return {"ref_latent": string_paths[0]}
            return {"ref_latents": string_paths}
        if self.mode == "audio":
            if len(string_paths) == 1:
                return {"ref_wav": string_paths[0]}
            return {"ref_wavs": string_paths}
        return {"no_ref": True}


class VoiceRegistry:
    """Discover MiRaiHead voices from files without a manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def discover(self) -> dict[str, VoiceSource]:
        root = self.ensure_root()
        embeddings: dict[str, VoiceSource] = {}
        folder_latents: dict[str, VoiceSource] = {}
        single_latents: dict[str, VoiceSource] = {}
        folder_audio: dict[str, VoiceSource] = {}
        single_audio: dict[str, VoiceSource] = {}

        for path in sorted(root.iterdir(), key=_natural_key):
            if path.name.startswith("."):
                continue
            if path.is_dir():
                latents = self._matching_files(path, LATENT_EXTENSIONS)
                audio = self._matching_files(path, AUDIO_EXTENSIONS)
                if latents:
                    folder_latents[path.name] = VoiceSource(path.name, "latents", latents)
                elif audio:
                    folder_audio[path.name] = VoiceSource(path.name, "audio", audio)
                continue
            if not path.is_file():
                continue

            lower_name = path.name.casefold()
            if lower_name.endswith(SPEAKER_EMBEDDING_SUFFIX):
                voice_id = path.name[: -len(SPEAKER_EMBEDDING_SUFFIX)]
                embeddings[voice_id] = VoiceSource(voice_id, "speaker_embedding", (path,))
            elif lower_name.endswith(SPEAKER_EMBEDDING_TYPO_SUFFIX):
                voice_id = path.name[: -len(SPEAKER_EMBEDDING_TYPO_SUFFIX)]
                canonical_path = self._canonical_embedding_copy(path, voice_id)
                embeddings[voice_id] = VoiceSource(
                    voice_id, "speaker_embedding", (canonical_path,)
                )
            elif path.suffix.casefold() in LATENT_EXTENSIONS:
                single_latents[path.stem] = VoiceSource(path.stem, "latents", (path,))
            elif path.suffix.casefold() in AUDIO_EXTENSIONS:
                single_audio[path.stem] = VoiceSource(path.stem, "audio", (path,))

        # Later updates have lower priority. This keeps the policy explicit:
        # embedding > folder latent set > single latent > folder audio > single audio.
        voices: dict[str, VoiceSource] = {}
        for candidates in (
            single_audio,
            folder_audio,
            single_latents,
            folder_latents,
            embeddings,
        ):
            voices.update(candidates)
        return voices

    def resolve(self, requested_voice_id: str | None) -> tuple[VoiceSource, bool]:
        requested = (requested_voice_id or "").strip()
        if requested.casefold() in NO_REF_IDS:
            return VoiceSource(requested or "none", "no_ref"), False

        voices = self.discover()
        exact = voices.get(requested)
        if exact is not None:
            return exact, False

        folded = requested.casefold()
        matches = [voice for voice_id, voice in voices.items() if voice_id.casefold() == folded]
        if len(matches) == 1:
            return matches[0], False

        LOGGER.warning("voice %r was not found; using no-reference inference", requested)
        return VoiceSource(requested, "no_ref"), True

    @staticmethod
    def _matching_files(directory: Path, extensions: set[str]) -> tuple[Path, ...]:
        return tuple(
            sorted(
                (
                    path.resolve()
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.casefold() in extensions
                ),
                key=_natural_key,
            )
        )

    def _canonical_embedding_copy(self, source: Path, voice_id: str) -> Path:
        cache_dir = self.root / ".mirai_voice_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{voice_id}{SPEAKER_EMBEDDING_SUFFIX}"
        source_stat = source.stat()
        if (
            not target.is_file()
            or target.stat().st_size != source_stat.st_size
            or target.stat().st_mtime_ns != source_stat.st_mtime_ns
        ):
            shutil.copy2(source, target)
        return target.resolve()


VOICES_DIR = Path(os.getenv("MIRAI_IRODORI_VOICES_DIR", "voices"))
VOICE_REGISTRY = VoiceRegistry(VOICES_DIR)


def _runtime_device(env_name: str) -> str:
    value = os.getenv(env_name, "auto").strip()
    return default_runtime_device() if value.casefold() in {"", "auto"} else value


def _split_hf_checkpoint_source(source: str) -> tuple[str, str | None]:
    raw = source.strip().strip("/")
    parts = raw.split("/")
    if not raw or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid Hugging Face checkpoint source: {source!r}")
    if len(parts) <= 2:
        return raw, None
    if len(parts) == 3:
        return "/".join(parts[:2]), parts[2]
    raise ValueError(
        "Hugging Face checkpoint subfolders must use owner/repo/subfolder format: "
        f"{source!r}"
    )


def _download_pinned_hf_checkpoint(source: str, revision: str) -> str:
    """Use a pinned cached snapshot, downloading it only when incomplete/missing."""
    from huggingface_hub import snapshot_download

    repo_id, subfolder = _split_hf_checkpoint_source(source)
    if subfolder is None:
        checkpoint_relative = Path("model.safetensors")
        allow_patterns = ["model.safetensors", "tokenizer/*"]
    else:
        checkpoint_relative = Path(subfolder) / "model.safetensors"
        allow_patterns = [
            checkpoint_relative.as_posix(),
            f"{subfolder}/tokenizer/*",
            "tokenizer/*",
        ]

    snapshot_kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "allow_patterns": allow_patterns,
    }
    snapshot_dir: Path | None = None
    try:
        snapshot_dir = Path(snapshot_download(**snapshot_kwargs, local_files_only=True))
    except Exception as exc:
        LOGGER.info(
            "Pinned Irodori snapshot %s is not complete in the local cache: %s",
            revision,
            exc,
        )

    if snapshot_dir is not None:
        checkpoint_path = snapshot_dir / checkpoint_relative
        if checkpoint_path.is_file():
            LOGGER.info("Using pinned Irodori snapshot from local cache: %s", snapshot_dir)
            return str(checkpoint_path)

    LOGGER.info("Downloading missing pinned Irodori snapshot: %s@%s", repo_id, revision)
    snapshot_dir = Path(snapshot_download(**snapshot_kwargs))
    checkpoint_path = snapshot_dir / checkpoint_relative
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face checkpoint source has no model.safetensors: {source}"
        )
    return str(checkpoint_path)


def _checkpoint_path() -> str:
    local_checkpoint = os.getenv("MIRAI_IRODORI_CHECKPOINT", "").strip()
    if local_checkpoint:
        path = Path(local_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint was not found: {path}")
        return str(path)

    hf_checkpoint = os.getenv(
        "MIRAI_IRODORI_HF_CHECKPOINT", DEFAULT_HF_CHECKPOINT
    ).strip()
    if not hf_checkpoint:
        raise ValueError(
            "Set MIRAI_IRODORI_CHECKPOINT or MIRAI_IRODORI_HF_CHECKPOINT."
        )
    revision = os.getenv("MIRAI_IRODORI_REVISION", "").strip()
    if revision:
        return _download_pinned_hf_checkpoint(hf_checkpoint, revision)

    # Preserve the official unpinned behavior for direct script users who do not
    # launch through start_mirai_server.bat.
    from irodori_tts.inference_runtime import download_hf_checkpoint

    return download_hf_checkpoint(hf_checkpoint)


@lru_cache(maxsize=1)
def get_runtime():
    key = RuntimeKey(
        checkpoint=_checkpoint_path(),
        model_device=_runtime_device("MIRAI_IRODORI_MODEL_DEVICE"),
        model_precision=os.getenv("MIRAI_IRODORI_MODEL_PRECISION", "fp32"),
        codec_repo=os.getenv(
            "MIRAI_IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim"
        ),
        codec_device=_runtime_device("MIRAI_IRODORI_CODEC_DEVICE"),
        codec_precision=os.getenv("MIRAI_IRODORI_CODEC_PRECISION", "fp32"),
        compile_model=_env_bool("MIRAI_IRODORI_COMPILE_MODEL"),
        compile_dynamic=_env_bool("MIRAI_IRODORI_COMPILE_DYNAMIC"),
    )
    runtime, _loaded = get_cached_runtime(key)
    return runtime


def _encode_wav(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
    if audio_cpu.ndim == 1:
        audio_data = audio_cpu.numpy()
    elif audio_cpu.shape[0] == 1:
        audio_data = audio_cpu.squeeze(0).numpy()
    else:
        audio_data = audio_cpu.transpose(0, 1).numpy()

    output = io.BytesIO()
    sf.write(output, audio_data, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _encode_voice_design_zip(result, elapsed: float) -> bytes:
    filenames = [f"candidate_{index:02d}.wav" for index in range(1, len(result.audios) + 1)]
    metadata = {
        "seed": result.used_seed,
        "candidates": filenames,
        "sample_rate": result.sample_rate,
        "inference_seconds": elapsed,
        "total_to_decode": result.total_to_decode,
        "stage_timings": [
            {"name": name, "seconds": seconds} for name, seconds in result.stage_timings
        ],
        "messages": list(result.messages),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, audio in zip(filenames, result.audios, strict=True):
            archive.writestr(filename, _encode_wav(audio, result.sample_rate))
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return output.getvalue()


async def _save_reference_uploads(
    uploads: list[UploadFile] | None,
    directory: Path,
) -> list[str]:
    paths: list[str] = []
    for index, upload in enumerate(uploads or (), start=1):
        suffix = Path(upload.filename or "").suffix.casefold()
        if suffix not in AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"reference audio {index} has an unsupported extension: {suffix or '(none)'}",
            )
        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail=f"reference audio {index} is empty")
        path = directory / f"reference_{index:03d}{suffix}"
        path.write_bytes(payload)
        paths.append(str(path))
    return paths


@asynccontextmanager
async def lifespan(_app: FastAPI):
    VOICE_REGISTRY.ensure_root()
    if _env_bool("MIRAI_IRODORI_PRELOAD"):
        await asyncio.to_thread(get_runtime)
    server_host = os.getenv("MIRAI_IRODORI_HOST", "0.0.0.0")
    server_port = int(os.getenv("MIRAI_IRODORI_PORT", "5000"))
    _print_connection_urls(server_host, server_port)
    yield


app = FastAPI(
    title="MiRaiHead Irodori-TTS v4 compatibility server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, object]:
    voices = VOICE_REGISTRY.discover()
    return {
        "status": "ok",
        "model": os.getenv(
            "MIRAI_IRODORI_HF_CHECKPOINT", DEFAULT_HF_CHECKPOINT
        ),
        "model_revision": os.getenv("MIRAI_IRODORI_REVISION", "") or None,
        "runtime_loaded": get_runtime.cache_info().currsize > 0,
        "voices_dir": str(VOICE_REGISTRY.root),
        "voices": len(voices),
    }


@app.get("/voices")
def list_voices() -> dict[str, object]:
    voices = VOICE_REGISTRY.discover()
    return {
        "voices": [
            {
                "id": voice.voice_id,
                "mode": voice.mode,
                "files": [str(path) for path in voice.paths],
            }
            for voice in sorted(voices.values(), key=lambda item: item.voice_id.casefold())
        ]
    }


@app.post("/tts")
async def synthesize(
    text: str = Form(...),
    speaker_id: str | None = Form(default=None),
    caption: str | None = Form(default=None),
    num_steps: int = Form(default=12),
    cfg_scale_text: float = Form(default=3.0),
    cfg_scale_speaker: float = Form(default=5.0),
    cfg_guidance_mode: str = Form(default="independent"),
    seed: int | None = Form(default=None),
    output_format: str = Form(default="wav", alias="format"),
) -> Response:
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if num_steps < 1 or num_steps > 120:
        raise HTTPException(status_code=400, detail="num_steps must be between 1 and 120")
    if output_format.casefold() != "wav":
        raise HTTPException(status_code=400, detail="only WAV output is supported")
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise HTTPException(status_code=400, detail="unsupported cfg_guidance_mode")

    voice, fell_back = VOICE_REGISTRY.resolve(speaker_id)
    request = SamplingRequest(
        text=clean_text,
        caption=(caption or "").strip() or None,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_guidance_mode=cfg_guidance_mode,
        seed=seed,
        **voice.sampling_kwargs(),
    )

    started = time.perf_counter()
    try:
        runtime = await asyncio.to_thread(get_runtime)
        result = await asyncio.to_thread(runtime.synthesize, request)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Irodori-TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - started
    duration = float(result.audio.shape[-1]) / float(result.sample_rate)
    wav_bytes = await asyncio.to_thread(_encode_wav, result.audio, result.sample_rate)
    headers = {
        "X-Inference-Time": f"{elapsed:.3f}",
        "X-Duration-Seconds": f"{duration:.3f}",
        "X-Seed": str(result.used_seed),
        "X-Irodori-Voice-Mode": voice.mode,
        "X-Irodori-Voice-Fallback": "true" if fell_back else "false",
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.post("/voice-design")
async def voice_design(
    text: str = Form(...),
    caption: str | None = Form(default=None),
    speaker_id: str | None = Form(default=None),
    reference_audio: list[UploadFile] | None = File(default=None),
    num_steps: int = Form(default=40),
    num_candidates: int = Form(default=1),
    seed: str | None = Form(default=None),
    seconds: str | None = Form(default=None),
    duration_scale: float = Form(default=1.0),
    t_schedule_mode: str = Form(default="linear"),
    sway_coeff: float = Form(default=-1.0),
    cfg_guidance_mode: str = Form(default="independent"),
    cfg_scale_text: float = Form(default=3.0),
    cfg_scale_caption: float = Form(default=4.0),
    cfg_scale_speaker: float = Form(default=5.0),
    cfg_scale: str | None = Form(default=None),
    cfg_min_t: float = Form(default=0.5),
    cfg_max_t: float = Form(default=1.0),
    context_kv_cache: bool = Form(default=True),
    speaker_kv_scale: str | None = Form(default=None),
    max_text_len: str | None = Form(default=None),
    max_caption_len: str | None = Form(default=None),
    truncation_factor: str | None = Form(default=None),
    rescale_k: str | None = Form(default=None),
    rescale_sigma: str | None = Form(default=None),
    lora_adapter: str | None = Form(default=None),
) -> Response:
    """Caption-conditioned inference for the standalone Voice Design GUI.

    This endpoint is intentionally separate from /tts so the deployed MiRaiHead
    contract remains unchanged. The response is a ZIP containing candidate WAVs
    and metadata.json.
    """
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if num_steps < 1 or num_steps > 120:
        raise HTTPException(status_code=400, detail="num_steps must be between 1 and 120")
    if num_candidates < 1 or num_candidates > MAX_VOICE_DESIGN_CANDIDATES:
        raise HTTPException(
            status_code=400,
            detail=f"num_candidates must be between 1 and {MAX_VOICE_DESIGN_CANDIDATES}",
        )
    if duration_scale <= 0:
        raise HTTPException(status_code=400, detail="duration_scale must be greater than zero")
    if t_schedule_mode not in {"linear", "sway"}:
        raise HTTPException(status_code=400, detail="unsupported t_schedule_mode")
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise HTTPException(status_code=400, detail="unsupported cfg_guidance_mode")

    parsed_seconds = _optional_float(seconds, "seconds")
    if parsed_seconds is not None and parsed_seconds <= 0:
        raise HTTPException(status_code=400, detail="seconds must be greater than zero")
    parsed_seed = _optional_int(seed, "seed")

    with tempfile.TemporaryDirectory(prefix="mirai-voice-design-") as temp_dir:
        ref_wavs = await _save_reference_uploads(reference_audio, Path(temp_dir))
        if ref_wavs:
            voice = VoiceSource("uploaded-reference", "audio", tuple(Path(p) for p in ref_wavs))
            fell_back = False
        else:
            voice, fell_back = VOICE_REGISTRY.resolve(speaker_id)

        no_ref = voice.mode == "no_ref"
        voice_ref_wavs = (
            ref_wavs if ref_wavs else [str(path) for path in voice.paths] if voice.mode == "audio" else []
        )
        request = SamplingRequest(
            text=clean_text,
            caption=_optional_text(caption),
            ref_wav=voice_ref_wavs[0] if len(voice_ref_wavs) == 1 else None,
            ref_wavs=voice_ref_wavs if len(voice_ref_wavs) > 1 else None,
            ref_latent=(str(voice.paths[0]) if voice.mode == "latents" and len(voice.paths) == 1 else None),
            ref_latents=(
                [str(path) for path in voice.paths]
                if voice.mode == "latents" and len(voice.paths) > 1
                else None
            ),
            ref_embed=(str(voice.paths[0]) if voice.mode == "speaker_embedding" else None),
            no_ref=no_ref,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=num_candidates,
            decode_mode="sequential",
            seconds=parsed_seconds,
            duration_scale=duration_scale,
            max_ref_seconds=None,
            max_text_len=_optional_int(max_text_len, "max_text_len"),
            max_caption_len=_optional_int(max_caption_len, "max_caption_len"),
            num_steps=num_steps,
            seed=parsed_seed,
            cfg_guidance_mode=cfg_guidance_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=0.0 if no_ref else cfg_scale_speaker,
            cfg_scale=_optional_float(cfg_scale, "cfg_scale"),
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=_optional_float(truncation_factor, "truncation_factor"),
            rescale_k=_optional_float(rescale_k, "rescale_k"),
            rescale_sigma=_optional_float(rescale_sigma, "rescale_sigma"),
            context_kv_cache=context_kv_cache,
            speaker_kv_scale=(
                None if no_ref else _optional_float(speaker_kv_scale, "speaker_kv_scale")
            ),
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            t_schedule_mode=t_schedule_mode,
            sway_coeff=sway_coeff,
            trim_tail=True,
            lora_adapter=_optional_text(lora_adapter),
        )

        started = time.perf_counter()
        try:
            runtime = await asyncio.to_thread(get_runtime)
            result = await asyncio.to_thread(runtime.synthesize, request)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Irodori-TTS voice design failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        elapsed = time.perf_counter() - started
        archive = await asyncio.to_thread(_encode_voice_design_zip, result, elapsed)

    headers = {
        "Content-Disposition": 'attachment; filename="voice-design.zip"',
        "X-Inference-Time": f"{elapsed:.3f}",
        "X-Seed": str(result.used_seed),
        "X-Irodori-Voice-Mode": voice.mode,
        "X-Irodori-Voice-Fallback": "true" if fell_back else "false",
        "X-Irodori-Candidates": str(len(result.audios)),
    }
    return Response(content=archive, media_type="application/zip", headers=headers)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("MIRAI_IRODORI_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server_host = os.getenv("MIRAI_IRODORI_HOST", "0.0.0.0")
    server_port = int(os.getenv("MIRAI_IRODORI_PORT", "5000"))
    uvicorn.run(
        app,
        host=server_host,
        port=server_port,
    )
