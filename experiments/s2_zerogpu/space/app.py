"""PresenceAI spike S2 Space (ADR 0007): faster-whisper large-v3-turbo on ZeroGPU.

Built into data/s2/space by experiments/s2_zerogpu/prepare_space.py, next to:
- audio.py: a copy of src/presenceai/speech/audio.py (ADR 0006: audio goes in as an array);
- s2_config.json: model subfolders, pinned model-repo revision, decoding options, reservation.

The model repo id comes from the Space variable S2_MODEL_REPO and is downloaded at start-up at the
pinned revision with the Space secret HF_TOKEN (read access to that one repo). Nothing is converted
here, nothing is stored, and no secret is printed.

Endpoints (called by experiments/s2_zerogpu/measure.py with gradio_client):
- gpu(wav, mode): mode "probe" checks that CTranslate2 sees the GPU and transcribes the clip;
  mode "full" transcribes it. Both run inside @spaces.GPU and report the time spent there.
- cpu(wav): Whisper small int8 on the Space CPU (fallback route; uses no GPU quota).
"""

import spaces  # noqa: I001  (must come before torch on ZeroGPU)
import torch  # loads the CUDA 12 cuBLAS / cuDNN 9 libraries that CTranslate2 also needs

import importlib.metadata as md
import json
import os
import time

import ctranslate2
import gradio as gr
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

import audio

with open(os.path.join(os.path.dirname(__file__), "s2_config.json"), encoding="utf-8") as _f:
    CFG = json.load(_f)
STARTED = time.time()
MODEL_ROOT = snapshot_download(
    os.environ["S2_MODEL_REPO"],
    revision=CFG["model_repo_revision"],
    token=os.environ.get("HF_TOKEN"),
)
TURBO = os.path.join(MODEL_ROOT, CFG["turbo_subfolder"])
SMALL = os.path.join(MODEL_ROOT, CFG["cpu_subfolder"])
DOWNLOAD_S = round(time.time() - STARTED, 3)
_CACHE: dict = {}


def _versions() -> dict:
    out = {}
    for pkg in (
        "torch",
        "ctranslate2",
        "faster-whisper",
        "av",
        "numpy",
        "gradio",
        "spaces",
        "huggingface_hub",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _text(segments) -> str:
    return " ".join(s.text.strip() for s in segments).strip()


@spaces.GPU(duration=CFG["gpu_duration_s"])
def gpu(wav: str, mode: str) -> str:
    t0 = time.perf_counter()
    info: dict = {"mode": mode, "torch_cuda_available": torch.cuda.is_available()}
    info["torch_cuda"] = torch.version.cuda
    info["torch_cudnn"] = torch.backends.cudnn.version()
    info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    info["ct2_cuda_devices"] = ctranslate2.get_cuda_device_count()
    if info["ct2_cuda_devices"] < 1:
        info["error"] = "CTranslate2 sees no CUDA device"
        info["gpu_s"] = round(time.perf_counter() - t0, 3)
        return json.dumps(info)
    info["ct2_cuda_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cuda"))
    info["model_cached"] = "turbo" in _CACHE
    t = time.perf_counter()
    if "turbo" not in _CACHE:
        _CACHE["turbo"] = WhisperModel(TURBO, device="cuda", compute_type=CFG["gpu_compute_type"])
    info["load_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    segments, tinfo = audio.transcribe(_CACHE["turbo"], wav, **CFG["transcribe"])
    info["transcribe_s"] = round(time.perf_counter() - t, 3)
    info["audio_s"] = round(tinfo.duration, 3)
    info["text"] = _text(segments)
    info["gpu_s"] = round(time.perf_counter() - t0, 3)
    return json.dumps(info)


def cpu(wav: str) -> str:
    t0 = time.perf_counter()
    model = WhisperModel(SMALL, device="cpu", compute_type=CFG["cpu_compute_type"])
    load_s = time.perf_counter() - t0
    t = time.perf_counter()
    segments, tinfo = audio.transcribe(model, wav, **CFG["transcribe"])
    return json.dumps(
        {
            "mode": "cpu_small",
            "load_s": round(load_s, 3),
            "transcribe_s": round(time.perf_counter() - t, 3),
            "audio_s": round(tinfo.duration, 3),
            "cpu_count": os.cpu_count(),
            "text": _text(segments),
        }
    )


def env() -> str:
    return json.dumps(
        {"versions": _versions(), "download_s": DOWNLOAD_S, "cpu_count": os.cpu_count()}
    )


with gr.Blocks(title="PresenceAI S2") as demo:
    wav_in = gr.Audio(type="filepath")
    mode_in = gr.Textbox(value="full")
    out = gr.Textbox()
    gr.Button("gpu").click(gpu, [wav_in, mode_in], out, api_name="gpu")
    gr.Button("cpu").click(cpu, [wav_in], out, api_name="cpu")
    gr.Button("env").click(env, [], out, api_name="env")

demo.queue(default_concurrency_limit=1).launch()
