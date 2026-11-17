"""C1 generator for Google Colab (ADR 0014). Runs as `python /content/generate_c1.py`.

colab.py build injects the config, the build info and the PresenceAI C1 code (schemas, synth,
evalconfig, configs/evaluation.yaml, resources/c1/scripts.yaml) and puts this script in a
notebook cell. Kokoro, misaki and their GPL dependencies are installed only here, never in the
app's environment (ADR 0014 §1).

espeak-ng is never loaded or called. Before Kokoro builds its pipelines, misaki's
`EspeakFallback` is replaced by `GuardFallback`:
- while checking the scripts, it records every word misaki's word lists do not cover;
- while synthesising, it raises.
Every script word is checked first; if any is missing, nothing is synthesised.

Mode (env C1_MODE): `timing` makes the first `timing_run_clips` clips (one per voice, each at a
different target rate) and projects the time for all; `full` makes every clip.
Outputs go to $C1_OUT (default /content/c1); the last line is `C1 DONE: status=ok|error`.
No secret is read and nothing is mounted: the model is public.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, ClassVar

_RAW_CONFIG = r"""__CONFIG_JSON__"""
_RAW_BUILD = r"""__BUILD_JSON__"""
CONFIG: dict[str, Any] = {} if _RAW_CONFIG.startswith("__") else json.loads(_RAW_CONFIG)
BUILD: dict[str, Any] = {} if _RAW_BUILD.startswith("__") else json.loads(_RAW_BUILD)
PACKAGE_B64 = "__PACKAGE_B64__"
OUT = Path(os.environ.get("C1_OUT", "/content/c1"))
REPO = Path(os.environ.get("C1_REPO", "/content/c1_repo"))
INITIAL_WPS = 2.7  # words per second of speech at speed 1; refined per voice after each clip


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)  # noqa: S603 (fixed commands, Colab only)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 (git's id)


def unpack_package() -> None:
    """The PresenceAI C1 code and configs, as built (checked by SHA-256 in the notebook)."""
    if PACKAGE_B64.startswith("__"):
        return  # local test: the package is already importable
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PACKAGE_B64))) as z:
        z.extractall(REPO)
    sys.path.insert(0, str(REPO / "src"))


# --- Colab-only set-up -----------------------------------------------------------------------


def install(gen: dict[str, Any], info: dict[str, Any]) -> None:
    sh([sys.executable, "-m", "pip", "install", "-q", *gen["pip"]])
    wheel = Path("/content") / gen["spacy_model"]["url"].rsplit("/", 1)[1]
    import urllib.request

    urllib.request.urlretrieve(gen["spacy_model"]["url"], wheel)  # noqa: S310 (pinned https URL)
    digest = sha256(wheel)
    pinned = gen["spacy_model"]["sha256"]
    info["spacy_model"] = {"file": wheel.name, "sha256": digest, "pinned": pinned}
    if pinned and digest != pinned:
        raise RuntimeError(f"{wheel.name}: SHA-256 {digest} differs from the pinned {pinned}")
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel)])


def download_kokoro(gen: dict[str, Any], info: dict[str, Any]) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths: dict[str, Path] = {}
    checks: dict[str, str] = {}
    for name, digest in {**gen["files"], **gen["git_blob_sha1"]}.items():
        p = Path(hf_hub_download(gen["model_id"], name, revision=gen["revision"]))
        got = git_blob_sha1(p) if name in gen["git_blob_sha1"] else sha256(p)
        if got != digest:
            raise RuntimeError(f"{name}: {got} differs from the pinned {digest}")
        paths[name] = p
        checks[name] = got
    info["model_files_verified"] = checks
    return paths


class GuardFallback:
    """Stands in for misaki's EspeakFallback: espeak-ng is never loaded or called."""

    strict: ClassVar[bool] = False
    calls: ClassVar[list[str]] = []

    def __init__(self, british: bool, version: Any = None) -> None:
        self.british = british

    def __call__(self, token: Any) -> tuple[None, None]:
        GuardFallback.calls.append(token.text)
        if GuardFallback.strict:
            raise RuntimeError(f"espeak-ng would be needed for {token.text!r}; stopping")
        return None, None


class KokoroTTS:
    sample_rate = 24_000

    def __init__(self, gen: dict[str, Any], voices: dict[str, str], paths: dict[str, Path]):
        import torch
        from kokoro import pipeline as kp
        from kokoro.model import KModel

        kp.espeak.EspeakFallback = GuardFallback  # before any pipeline is built
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model = (
            KModel(
                repo_id=gen["model_id"],
                config=str(paths["config.json"]),
                model=str(paths["kokoro-v1_0.pth"]),
            )
            .to(device)
            .eval()
        )
        self.pipes = {
            code: kp.KPipeline(lang_code=code, repo_id=gen["model_id"], model=model)
            for code in sorted(set(voices.values()))
        }
        for pipe in self.pipes.values():
            if not isinstance(pipe.g2p.fallback, GuardFallback):
                raise RuntimeError("espeak guard not in place")
        self.voices = voices
        self.paths = paths

    def g2p(self, code: str) -> Any:
        return self.pipes[code].g2p

    def synth(self, text: str, voice: str, speed: float) -> Any:
        import numpy as np

        pipe = self.pipes[self.voices[voice]]
        parts = []
        for r in pipe(text, voice=str(self.paths[f"voices/{voice}.pt"]), speed=speed,
                      split_pattern=None):  # fmt: skip
            for tk in r.tokens or []:
                if tk.text.isalpha() and (tk._ is None or tk._.rating is None):
                    raise RuntimeError(f"{tk.text!r} has no word-list rating; stopping")
            parts.append(r.audio.detach().cpu().numpy().astype(np.float32))
        if not parts:
            raise RuntimeError(f"no audio for {text!r}")
        return np.concatenate(parts)


# --- generation (also run locally in tests with a fake TTS) ----------------------------------


def check_words(scripts: Any, cfg: Any, g2p_for: Any) -> dict[str, Any]:
    """Run misaki's G2P over every phrase and filler; record each word's lowest rating."""
    words: dict[str, dict[str, int | None]] = {}
    missing: dict[str, list[str]] = {}
    for code in sorted(set(cfg.voices.values())):
        g2p = g2p_for(code)
        rated: dict[str, int | None] = {}
        before = len(GuardFallback.calls)
        texts = [p.text for s in scripts if cfg.voices[s.voice] == code for p in s.phrases]
        texts += list(cfg.fillers.tokens)
        for text in texts:
            _, tokens = g2p(text)
            for tk in tokens:
                if not tk.text.isalpha():
                    continue
                r = tk._.rating if tk._ is not None else None
                w = tk.text.lower()
                old = rated.get(w, 99)
                rated[w] = None if r is None or old is None else min(int(r), int(old))
        needed = {w.lower() for t in texts for w in re.findall(r"[A-Za-z]+", t)}
        for w in needed - set(rated):
            rated[w] = None
        fell_back = {w.lower() for w in GuardFallback.calls[before:]}
        missing[code] = sorted({w for w, r in rated.items() if r is None} | fell_back)
        words[code] = {w: (None if w in fell_back else r) for w, r in sorted(rated.items())}
    return {"words": words, "missing": missing}


def select(scripts: Any, cfg: Any, mode: str) -> list[tuple[int, int]]:
    pairs = [(i, r) for i in range(len(scripts)) for r in range(cfg.renders_per_script)]
    if mode == "full":
        return pairs
    first_per_voice: dict[str, int] = {}
    for i, s in enumerate(scripts):
        first_per_voice.setdefault(s.voice, i)
    picks = list(first_per_voice.values())[: cfg.timing_run_clips]
    return [(i, k % cfg.renders_per_script) for k, i in enumerate(picks)]


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, round(q * (len(ys) - 1)))]


def generate(tts: Any, mode: str, out: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    from presenceai.evalconfig import REPO_ROOT, load_evaluation
    from presenceai.synth import c1
    from presenceai.synth import scripts as sc

    cfg = load_evaluation().c1
    scripts = sc.load_scripts(REPO_ROOT / cfg.scripts)
    problems = sc.check_set(scripts, cfg)
    if problems:
        raise RuntimeError("script checks failed: " + "; ".join(problems[:10]))
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    wps: dict[str, float] = {}
    clips: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    lead: list[float] = []
    tail: list[float] = []
    silence: list[float] = []
    internal: list[float] = []
    for i, render in select(scripts, cfg, mode):
        s = scripts[i]
        target = cfg.target_wpm[render]
        plan = c1.plan_clip(s, render, target, c1.clip_rng(cfg, i, render), cfg)
        t0 = time.perf_counter()
        res = c1.render(s, plan, tts, cfg, wps.get(s.voice, INITIAL_WPS))
        wps[s.voice] = res.wps_at_speed_1
        seconds = time.perf_counter() - t0
        if res.truth is None or res.clean is None or res.stats is None:
            dropped.append({"clip_id": plan.clip_id, "reason": res.dropped, "seconds": seconds})
            continue
        audio = c1.add_noise_floor(res.clean, cfg.noise_floor_dbfs, c1.clip_rng(cfg, i, render))
        sf.write(out / "clips" / f"{plan.clip_id}.flac", np.clip(audio, -1, 1), tts.sample_rate,
                 subtype="PCM_16")  # fmt: skip
        (out / "truth" / f"{plan.clip_id}.json").write_text(
            res.truth.model_dump_json(indent=1), encoding="utf-8"
        )
        lead += res.stats.trimmed_lead_s
        tail += res.stats.trimmed_tail_s
        silence += res.stats.silence_db_re_peak
        internal += [x.duration_s for x in res.truth.internal_silences]
        clips.append({
            "clip_id": plan.clip_id, "voice": s.voice, "target_wpm": target,
            "wpm": round(res.truth.wpm, 2), "attempts": res.attempts,
            "speed": round(res.truth.speed, 4), "audio_s": round(res.truth.duration_s, 3),
            "seconds": round(seconds, 3), "phrases": res.stats.phrases,
            "kokoro_pauses": sum(1 for p in res.truth.pauses if p.source == "kokoro"),
            "fillers": res.truth.fillers,
        })  # fmt: skip
    total = sum(c["seconds"] for c in clips) + sum(d["seconds"] for d in dropped)
    n_all = len(scripts) * cfg.renders_per_script
    made = len(clips) + len(dropped)
    return {
        "clips": clips,
        "dropped": dropped,
        "generation_s": round(total, 2),
        "audio_s": round(sum(c["audio_s"] for c in clips), 2),
        "projected_all_clips_s": round(total / made * n_all, 1) if made else None,
        "trim": {
            "threshold_db_below_peak": cfg.trim.threshold_db_below_peak,
            "lead_s_p50_p95": [_pct(lead, 0.5), _pct(lead, 0.95)],
            "tail_s_p50_p95": [_pct(tail, 0.5), _pct(tail, 0.95)],
            "trimmed_level_db_re_peak_p50_p95_max": [
                _pct(silence, 0.5),
                _pct(silence, 0.95),
                max(silence) if silence else None,
            ],
        },
        "internal_silences": {
            "count": len(internal),
            "at_or_above_join_s": sum(1 for x in internal if x >= cfg.trim.internal_joins_truth_s),
            "durations_s_p50_max": [_pct(internal, 0.5), max(internal) if internal else None],
        },
    }


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("kokoro", "misaki", "torch", "spacy", "en_core_web_sm", "numpy", "soundfile",
                "phonemizer-fork", "espeakng-loader", "pydantic"):  # fmt: skip
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def finish(info: dict[str, Any], out: Path) -> None:
    table = {
        p.relative_to(out).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "run_info.json"
    }
    info["sha256"] = table
    info["finished_unix"] = time.time()
    (out / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"C1 DONE: status={info['status']}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("C1_MODE", BUILD.get("mode", "timing"))
    info: dict[str, Any] = {"build": BUILD, "mode": mode, "started_unix": time.time()}
    try:
        unpack_package()
        gen = CONFIG["generator"]
        t0 = time.perf_counter()
        install(gen, info)
        paths = download_kokoro(gen, info)
        info["setup_s"] = round(time.perf_counter() - t0, 1)
        info["versions"] = versions()
        t0 = time.perf_counter()
        tts = KokoroTTS(gen, CONFIG["voices"], paths)
        info["model_load_s"] = round(time.perf_counter() - t0, 1)
        info["device"] = tts.device
        if tts.device == "cuda":
            import torch

            info["gpu"] = torch.cuda.get_device_name(0)

        from presenceai.evalconfig import REPO_ROOT, load_evaluation
        from presenceai.synth import scripts as sc

        cfg = load_evaluation().c1
        scripts = sc.load_scripts(REPO_ROOT / cfg.scripts)
        check = check_words(scripts, cfg, tts.g2p)
        (OUT / "lexicon_check.json").write_text(
            json.dumps({**check, "misaki": info["versions"].get("misaki")}, indent=1),
            encoding="utf-8",
        )
        missing = {k: v for k, v in check["missing"].items() if v}
        if missing:
            info["status"] = "error"
            info["error"] = f"words not in misaki's word lists: {missing}"
            return 1
        GuardFallback.strict = True
        info["generation"] = generate(tts, mode, OUT)
        info["espeak_guard_calls"] = len(GuardFallback.calls)
        info["status"] = "ok"
        return 0
    except Exception as e:  # recorded in run_info.json, then re-raised as exit code 1
        info["status"] = "error"
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()
        return 1
    finally:
        finish(info, OUT)


if __name__ == "__main__":
    raise SystemExit(main())
