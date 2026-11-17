"""Speech recognition: faster-whisper -> `Transcript` with word timestamps and probabilities.

- Models are loaded only from local folders under `models/` (`dir` in configs/models.yaml
  `asr:`), with `local_files_only=True`. Nothing is downloaded here; a missing model raises
  `ModelNotAvailable` naming the `scripts/download_models.py` command (AGENTS.md rule 2).
- Audio always reaches the model as an array through `audio.transcribe` (ADR 0006).
- faster-whisper's own VAD filter is off: pauses are measured by our VAD step (P1 slice 2), and
  filtering here would remove the very silences we measure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from presenceai.schemas.speech import AsrQuality, Segment, SpeechLanguage, Transcript, Word
from presenceai.settings import MODELS_CONFIG, REPO_ROOT, AsrSettings
from presenceai.speech import audio

MODEL_NAMES = ("whisper_small", "whisper_turbo")
MODELS_ROOT = REPO_ROOT / "models"
# Files a CTranslate2 Whisper folder needs (written by ct2-transformers-converter --copy_files).
REQUIRED_FILES = ("config.json", "model.bin", "vocabulary.json", "tokenizer.json")


class ModelNotAvailable(RuntimeError):
    """The requested ASR model is unknown or not downloaded."""


def model_dir(
    name: str, *, models_yaml: Path = MODELS_CONFIG, models_root: Path = MODELS_ROOT
) -> Path:
    entries = (yaml.safe_load(models_yaml.read_text(encoding="utf-8")) or {}).get("asr") or {}
    entry = entries.get(name)
    if not isinstance(entry, dict) or not entry.get("dir"):
        raise ModelNotAvailable(f"unknown ASR model {name!r}; choose one of {sorted(entries)}")
    return models_root / str(entry["dir"])


def is_verbatim(name: str, *, models_yaml: Path = MODELS_CONFIG) -> bool:
    """`verbatim` flag of an `asr:` entry: True only for a model that keeps fillers (T1)."""
    entries = (yaml.safe_load(models_yaml.read_text(encoding="utf-8")) or {}).get("asr") or {}
    entry = entries.get(name)
    return bool(entry.get("verbatim")) if isinstance(entry, dict) else False


def _whisper_model_class() -> Any:
    from faster_whisper import WhisperModel  # heavy import, only when a model is really loaded

    return WhisperModel


def load_model(
    name: str,
    options: AsrSettings,
    *,
    models_yaml: Path = MODELS_CONFIG,
    models_root: Path = MODELS_ROOT,
) -> Any:
    folder = model_dir(name, models_yaml=models_yaml, models_root=models_root)
    missing = [f for f in REQUIRED_FILES if not (folder / f).is_file()]
    if missing:
        raise ModelNotAvailable(
            f"{name} is not downloaded ({folder.name}/ lacks {', '.join(missing)}); run: "
            f"uv run python scripts/download_models.py --only {name}"
        )
    return _whisper_model_class()(
        str(folder),
        device=options.device,
        compute_type=options.compute_type,
        local_files_only=True,
    )


def transcribe_file(
    model: Any, path: str | Path, *, language: SpeechLanguage, options: AsrSettings
) -> Transcript:
    raw_segments, info = audio.transcribe(
        model,
        path,
        language=language,
        word_timestamps=True,
        vad_filter=False,
        beam_size=options.beam_size,
        temperature=options.temperature,
        condition_on_previous_text=options.condition_on_previous_text,
    )
    text = ""
    segments: list[Segment] = []
    for seg in raw_segments:
        words: list[Word] = []
        for w in seg.words or ():
            token = str(w.word).strip()
            if not token:
                continue
            if text:
                text += " "
            start_char = len(text)
            text += token
            start_s = max(0.0, float(w.start))
            words.append(
                Word(
                    text=token,
                    start_s=start_s,
                    end_s=max(start_s, float(w.end)),
                    probability=min(1.0, max(0.0, float(w.probability))),
                    start_char=start_char,
                    end_char=len(text),
                )
            )
        if words:
            seg_start = max(0.0, float(seg.start))
            segments.append(
                Segment(
                    start_s=seg_start,
                    end_s=max(seg_start, float(seg.end)),
                    words=tuple(words),
                    avg_logprob=float(seg.avg_logprob),
                    no_speech_prob=min(1.0, max(0.0, float(seg.no_speech_prob))),
                )
            )
    return Transcript(
        language=language,
        duration_s=max(0.0, float(info.duration)),
        text=text,
        segments=tuple(segments),
    )


def quality(transcript: Transcript) -> AsrQuality:
    words, segs = transcript.words, transcript.segments
    return AsrQuality(
        word_count=len(words),
        mean_word_probability=sum(w.probability for w in words) / len(words) if words else None,
        mean_no_speech_prob=sum(s.no_speech_prob for s in segs) / len(segs) if segs else None,
    )
