"""C1 synthetic speech: clips built from separately generated phrases (ADR 0014 §3).

- Each phrase is one TTS call, trimmed to its speech: the first and last 5 ms frame within
  `threshold_db_below_peak` of the phrase's loudest frame, then refined to the first and last
  sample above that level. Quiet onsets and endings (s, f, h) are 30-45 dB below a vowel, so
  they are kept.
- Phrases are joined by silences of exact length from the clip plan, so every true pause and
  its position come from the construction. Silences the TTS made inside a phrase are found the
  same way and reported; from `internal_joins_truth_s` they are true pauses too.
- The true speaking rate is measured on the finished clean audio: words / (last speech offset -
  first speech onset), the definition of `rate.wpm` (ADR 0011). The TTS speed setting is only a
  means to reach the target rate.
- The TTS is passed in (`PhraseTTS`), so this module never imports a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from presenceai.evalconfig import C1Settings, C1Trim
from presenceai.schemas.c1 import ClipPlan, ClipTruth, GapPlan, InternalSilence, Script, TruePause
from presenceai.schemas.speech import PausePosition

_EPS = 1e-20


class PhraseTTS(Protocol):
    sample_rate: int

    def synth(self, text: str, voice: str, speed: float) -> np.ndarray: ...


# --- trimming -------------------------------------------------------------------------------


def _frames(x: np.ndarray, sr: int, cfg: C1Trim) -> tuple[np.ndarray, int]:
    n = max(1, round(sr * cfg.frame_ms / 1000))
    count = math.ceil(x.size / n)
    padded = np.zeros(count * n, dtype=np.float64)
    padded[: x.size] = x
    levels = 10 * np.log10(np.mean(padded.reshape(count, n) ** 2, axis=1) + _EPS)
    return levels, n


def _speech(x: np.ndarray, sr: int, cfg: C1Trim) -> tuple[np.ndarray, int, float] | None:
    """(speech-frame mask, frame length, sample threshold) or None for a silent input."""
    if x.size == 0 or not np.any(x):
        return None
    levels, n = _frames(x, sr, cfg)
    floor_db = float(levels.max()) - cfg.threshold_db_below_peak
    return levels >= floor_db, n, 10 ** (floor_db / 20)


def _first_above(x: np.ndarray, lo: int, hi: int, thr: float) -> int:
    idx = np.flatnonzero(np.abs(x[lo:hi]) >= thr)
    return lo + int(idx[0]) if idx.size else hi


def _last_above(x: np.ndarray, lo: int, hi: int, thr: float) -> int:
    idx = np.flatnonzero(np.abs(x[lo:hi]) >= thr)
    return lo + int(idx[-1]) + 1 if idx.size else lo


def trim(x: np.ndarray, sr: int, cfg: C1Trim) -> tuple[int, int] | None:
    """Sample range [start, end) of the speech in one phrase; None if it is silent."""
    found = _speech(x, sr, cfg)
    if found is None:
        return None
    mask, n, thr = found
    idx = np.flatnonzero(mask)
    a = int(idx[0]) * n
    b = min(x.size, (int(idx[-1]) + 1) * n)
    # refine to the sample: the first and last sample at the threshold level, searched one
    # frame either side of the frame edge
    start = _first_above(x, max(0, a - n), min(x.size, a + n), thr)
    end = _last_above(x, max(0, b - n), min(x.size, b + n), thr)
    return (start, end) if end > start else (a, b)


def internal_silences(x: np.ndarray, sr: int, cfg: C1Trim) -> list[tuple[int, int]]:
    """Silences of at least `internal_min_silence_s` between a phrase's first and last speech."""
    found = _speech(x, sr, cfg)
    if found is None:
        return []
    mask, n, thr = found
    idx = np.flatnonzero(mask)
    inner = mask[idx[0] : idx[-1] + 1]
    min_frames = math.ceil(cfg.internal_min_silence_s * sr / n) - 1
    out: list[tuple[int, int]] = []
    padded = np.concatenate([[True], inner, [True]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for f0, f1 in zip(edges[::2], edges[1::2], strict=True):
        if f1 - f0 < min_frames:
            continue
        lo, hi = (int(idx[0]) + int(f0)) * n, (int(idx[0]) + int(f1)) * n
        a = _last_above(x, max(0, lo - n), lo + n, thr)
        b = _first_above(x, max(0, hi - n), min(x.size, hi + n), thr)
        if (b - a) / sr >= cfg.internal_min_silence_s:
            out.append((a, b))
    return out


# --- planning -------------------------------------------------------------------------------


def clip_rng(cfg: C1Settings, script_index: int, render: int) -> np.random.Generator:
    return np.random.default_rng([cfg.seed, script_index, render])


def _draw(rng: np.random.Generator, lo_hi: tuple[float, float], log: bool) -> float:
    lo, hi = lo_hi
    v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if log else rng.uniform(lo, hi)
    return round(float(v), 3)


def plan_clip(
    script: Script, render: int, target_wpm: float, rng: np.random.Generator, cfg: C1Settings
) -> ClipPlan:
    g = cfg.gaps
    raw: list[tuple[str, float]] = []
    for p in script.phrases[:-1]:
        kind = p.boundary_after
        if kind == "end":
            raise ValueError(f"{script.id}: only the last phrase may end the script")
        if rng.random() < g.pause_probability[kind]:
            raw.append((kind, _draw(rng, g.pause_s, g.pause_distribution == "log_uniform")))
        else:
            raw.append((kind, _draw(rng, g.short_s, False)))
    fillers: dict[int, tuple[str, float]] = {}
    f = cfg.fillers
    if raw and rng.random() < f.clip_share:
        n = int(rng.integers(f.per_clip[0], f.per_clip[1] + 1))
        for i in sorted(rng.choice(len(raw), size=min(n, len(raw)), replace=False).tolist()):
            fillers[int(i)] = (str(rng.choice(list(f.tokens))), _draw(rng, f.after_s, False))
    gaps = tuple(
        GapPlan(
            after_phrase=i,
            boundary=kind,  # type: ignore[arg-type]
            seconds=sec,
            filler=fillers[i][0] if i in fillers else None,
            after_filler_s=fillers[i][1] if i in fillers else None,
        )
        for i, (kind, sec) in enumerate(raw)
    )
    return ClipPlan(
        clip_id=f"{script.id}_r{render}",
        script_id=script.id,
        voice=script.voice,
        render=render,
        target_wpm=target_wpm,
        gaps=gaps,
    )


# --- assembling -----------------------------------------------------------------------------


@dataclass
class AssemblyStats:
    """What the trimmer saw, to re-check the threshold on real TTS output (ADR 0014)."""

    phrases: int = 0
    trimmed_lead_s: list[float] = field(default_factory=list)
    trimmed_tail_s: list[float] = field(default_factory=list)
    silence_db_re_peak: list[float] = field(default_factory=list)  # level of what was trimmed


def _silence_level(x: np.ndarray, a: int, b: int, sr: int, cfg: C1Trim) -> float | None:
    rest = np.concatenate([x[:a], x[b:]]).astype(np.float64)
    if rest.size < sr * cfg.frame_ms / 1000:
        return None
    peak = float(_frames(x, sr, cfg)[0].max())
    return float(10 * np.log10(np.mean(rest**2) + _EPS)) - peak


def assemble(
    script: Script, plan: ClipPlan, tts: PhraseTTS, speed: float, cfg: C1Settings
) -> tuple[np.ndarray, ClipTruth, AssemblyStats]:
    """Clean clip (no noise floor), its truth, and trimming statistics."""
    sr = tts.sample_rate
    if len(plan.gaps) != len(script.phrases) - 1 or any(
        g.after_phrase != i for i, g in enumerate(plan.gaps)
    ):
        raise ValueError(f"{plan.clip_id}: the plan does not fit the script")
    stats = AssemblyStats()
    parts: list[np.ndarray] = []
    cursor = 0
    spans: list[tuple[int, int]] = []
    pauses: list[TruePause] = []
    internal: list[InternalSilence] = []

    def silence(seconds: float) -> tuple[int, int]:
        nonlocal cursor
        n = round(seconds * sr)
        parts.append(np.zeros(n, np.float32))
        start, cursor = cursor, cursor + n
        return start, cursor

    def speak(text: str, phrase_index: int) -> None:
        nonlocal cursor
        x = np.asarray(tts.synth(text, script.voice, speed), dtype=np.float32)
        bounds = trim(x, sr, cfg.trim)
        if bounds is None:
            raise ValueError(f"{plan.clip_id}: the TTS returned silence for {text!r}")
        a, b = bounds
        stats.phrases += 1
        stats.trimmed_lead_s.append(a / sr)
        stats.trimmed_tail_s.append((x.size - b) / sr)
        level = _silence_level(x, a, b, sr, cfg.trim)
        if level is not None:
            stats.silence_db_re_peak.append(level)
        for s0, s1 in internal_silences(x, sr, cfg.trim):
            internal.append(
                InternalSilence(
                    phrase=phrase_index,
                    start_s=(cursor + s0 - a) / sr,
                    end_s=(cursor + s1 - a) / sr,
                )
            )
        parts.append(x[a:b])
        spans.append((cursor, cursor + (b - a)))
        cursor += b - a

    silence(cfg.edge_silence_s)
    words = script.words
    fillers = 0
    for i, phrase in enumerate(script.phrases):
        speak(phrase.text, i)
        if i == len(script.phrases) - 1:
            break
        gap = plan.gaps[i]
        start, end = silence(gap.seconds)
        if gap.seconds >= cfg.gaps.pause_s[0]:
            pauses.append(
                TruePause(
                    start_s=start / sr,
                    end_s=end / sr,
                    position=PausePosition(gap.boundary),
                    source="inserted",
                )
            )
        if gap.filler is not None and gap.after_filler_s is not None:
            speak(gap.filler, i)
            fillers += 1
            words += 1
            silence(gap.after_filler_s)
    silence(cfg.edge_silence_s)

    joins = cfg.trim.internal_joins_truth_s
    pauses += [
        TruePause(
            start_s=s.start_s,
            end_s=s.end_s,
            position=PausePosition.MID_SENTENCE,
            source="kokoro",
        )
        for s in internal
        if s.duration_s >= joins
    ]
    pauses.sort(key=lambda p: p.start_s)
    clean = np.concatenate(parts)
    onset, offset = spans[0][0] / sr, spans[-1][1] / sr
    speech_s = sum(b - a for a, b in spans) / sr - sum(
        s.duration_s for s in internal if s.duration_s >= joins
    )
    truth = ClipTruth(
        clip_id=plan.clip_id,
        script_id=script.id,
        voice=script.voice,
        render=plan.render,
        sample_rate=sr,
        duration_s=clean.size / sr,
        words=words,
        fillers=fillers,
        speech_onset_s=onset,
        speech_offset_s=offset,
        speech_spans=tuple((a / sr, b / sr) for a, b in spans),
        wpm=words * 60 / (offset - onset),
        articulation_wpm=words * 60 / speech_s,
        target_wpm=plan.target_wpm,
        speed=speed,
        attempts=1,
        pauses=tuple(pauses),
        internal_silences=tuple(internal),
        text=script.text,
    )
    return clean, truth, stats


# --- rendering to a target rate -------------------------------------------------------------


@dataclass
class RenderResult:
    truth: ClipTruth | None
    clean: np.ndarray | None
    stats: AssemblyStats | None
    attempts: int
    wps_at_speed_1: float  # updated estimate for this voice (words per second of speech)
    dropped: str | None = None


def _speed_for(words: int, target: float, gaps_s: float, wps1: float, cfg: C1Settings) -> float:
    lo, hi = cfg.speed_range
    speech_s = words * 60 / target - gaps_s
    if speech_s <= 0:
        return hi
    return float(min(hi, max(lo, words / (wps1 * speech_s))))


def render(
    script: Script, plan: ClipPlan, tts: PhraseTTS, cfg: C1Settings, wps_at_speed_1: float
) -> RenderResult:
    """Assemble at a speed that reaches the plan's target rate, re-rendering with a corrected
    speed up to `max_rerenders` times. The clip is dropped if its measured rate stays outside
    `wpm_range` or its length outside `clip_s`."""
    gaps_s = sum(g.seconds + (g.after_filler_s or 0.0) for g in plan.gaps)
    words = script.words + sum(1 for g in plan.gaps if g.filler)
    wps1 = wps_at_speed_1
    speed = _speed_for(words, plan.target_wpm, gaps_s, wps1, cfg)
    attempts = 0
    while True:
        attempts += 1
        clean, truth, stats = assemble(script, plan, tts, speed, cfg)
        speaking = truth.speech_offset_s - truth.speech_onset_s - gaps_s
        if speaking > 0:
            wps1 = words / (speaking * speed)
        if abs(truth.wpm - plan.target_wpm) <= cfg.accept_within_wpm:
            break
        if attempts > cfg.max_rerenders:
            break
        speed = _speed_for(words, plan.target_wpm, gaps_s, wps1, cfg)
    truth = truth.model_copy(update={"attempts": attempts})
    lo, hi = cfg.wpm_range
    if not lo <= truth.wpm <= hi:
        reason = f"measured {truth.wpm:.0f} wpm, outside {lo:g}-{hi:g} after {attempts} renders"
        return RenderResult(None, None, None, attempts, wps1, dropped=reason)
    c_lo, c_hi = cfg.clip_s
    if not c_lo <= truth.duration_s <= c_hi:
        reason = f"clip is {truth.duration_s:.1f} s, outside {c_lo:g}-{c_hi:g} s"
        return RenderResult(None, None, None, attempts, wps1, dropped=reason)
    return RenderResult(truth, clean, stats, attempts, wps1)


# --- noise ----------------------------------------------------------------------------------


def add_noise_floor(x: np.ndarray, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """White noise at `dbfs` (RMS) over the whole clip, so pauses are not digital silence."""
    noise = rng.standard_normal(x.size) * 10 ** (dbfs / 20)
    return (x.astype(np.float64) + noise).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS noise whose power falls 3 dB per octave."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / np.sqrt(np.mean(x**2))


def span_mask(spans: tuple[tuple[float, float], ...], sr: int, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for a, b in spans:
        mask[round(a * sr) : round(b * sr)] = True
    return mask


def mix_noise(
    x: np.ndarray,
    spans: tuple[tuple[float, float], ...],
    sr: int,
    snr_db: float,
    rng: np.random.Generator,
    color: str,
) -> np.ndarray:
    """Add noise so that speech power (over the true speech spans) / noise power = `snr_db`."""
    xs = x.astype(np.float64)
    p_speech = float(np.mean(xs[span_mask(spans, sr, x.size)] ** 2))
    noise = pink_noise(x.size, rng) if color == "pink" else rng.standard_normal(x.size)
    noise = noise / np.sqrt(np.mean(noise**2))
    return (xs + noise * math.sqrt(p_speech / 10 ** (snr_db / 10))).astype(np.float32)


def rate_band(wpm: float, bands: tuple[tuple[float, float], ...]) -> str | None:
    for i, (lo, hi) in enumerate(bands):
        last = i == len(bands) - 1
        if lo <= wpm < hi or (last and wpm == hi):
            return f"{lo:g}-{hi:g}"
    return None
