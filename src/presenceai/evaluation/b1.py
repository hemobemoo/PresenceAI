"""B1: the owner's annotated recordings as the real-speech check (ADR 0016).

- `prefill` turns an app report into the two annotation files, so the owner only corrects:
  the transcript and speech start/end in `<clip>.yaml`, the pauses in `<clip>.labels.txt`
  (an Audacity label track);
- `score_clip` compares a corrected annotation with a report, using the E2 matching rules
  unchanged (`evaluation.e2`, ADR 0014 §6). B1 is descriptive: no pass/fail, no tuning target;
- clip names and the clip plan follow `docs/data/B1_RECORDING_GUIDE.md`.

Results hold numbers only; transcript text never leaves data/b1/.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import yaml

from presenceai.evalconfig import EvaluationSettings
from presenceai.evaluation import e2
from presenceai.schemas.b1 import (
    LABELS,
    UNDECIDED,
    B1Annotation,
    B1ClipScore,
    ClipType,
    LabelledPause,
    Prefill,
    RemovedWord,
)
from presenceai.schemas.c1 import TruePause, f1
from presenceai.schemas.speech import PausePosition, SpeechLanguage, SpeechReport
from presenceai.settings import PauseSettings
from presenceai.speech import fillers, pauses
from presenceai.text.normalize import normalize_token

# number -> (type, Arabic code-switching, Arabic long sentences); docs/data/B1_RECORDING_GUIDE.md
CLIP_PLAN: dict[int, tuple[ClipType, bool, bool]] = {
    1: ("normal", False, False),
    2: ("normal", True, False),
    3: ("normal", False, True),
    4: ("normal", False, True),
    5: ("defense", True, False),
    6: ("defense", True, True),
    7: ("slow", False, False),
    8: ("fast", False, False),
    9: ("fillers", False, False),
    10: ("noise", False, False),
}
LANGS: tuple[SpeechLanguage, ...] = ("en", "ar")
AUDIO_SUFFIXES = (".wav", ".flac")
_NAME = re.compile(r"^b1_(en|ar)_(\d{2})_([a-z]+)$")
_LABEL_OF = {position: label for label, position in LABELS.items()}


def clip_stem(language: SpeechLanguage, number: int) -> str:
    return f"b1_{language}_{number:02d}_{CLIP_PLAN[number][0]}"


def expected_stems() -> list[str]:
    return [clip_stem(lang, n) for lang in LANGS for n in CLIP_PLAN]


def parse_stem(stem: str) -> tuple[SpeechLanguage, int]:
    """'b1_ar_07_slow' -> ("ar", 7); anything off the plan raises ValueError."""
    m = _NAME.match(stem)
    if not m:
        raise ValueError(f"{stem}: name it b1_<en|ar>_<01-10>_<type>, e.g. b1_en_01_normal")
    lang: SpeechLanguage = "en" if m.group(1) == "en" else "ar"
    number = int(m.group(2))
    if number not in CLIP_PLAN:
        raise ValueError(f"{stem}: clip numbers run from 01 to {len(CLIP_PLAN):02d}")
    if m.group(3) != CLIP_PLAN[number][0]:
        raise ValueError(f"{stem}: clip {number:02d} must be named {clip_stem(lang, number)}")
    return lang, number


# --- annotation files -------------------------------------------------------------------------


def write_labels(items: Sequence[LabelledPause]) -> str:
    """Audacity label track: one `start<TAB>end<TAB>label` line per pause."""
    return "".join(f"{p.start_s:.6f}\t{p.end_s:.6f}\t{p.label}\n" for p in items)


def read_labels(content: str) -> tuple[LabelledPause, ...]:
    out: list[LabelledPause] = []
    for n, line in enumerate(content.splitlines(), start=1):
        if not line.strip() or line.startswith("\\"):  # Audacity frequency lines start with "\"
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"labels line {n}: expected start, end and label separated by tabs")
        start, end, label = float(parts[0]), float(parts[1]), parts[2].strip()
        out.append(LabelledPause(start_s=start, end_s=end, label=label))
    return tuple(sorted(out, key=lambda p: p.start_s))


def _str_presenter(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data or len(data) > 70 else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, _str_presenter)

ANNOTATION_HEADER = """\
# B1 annotation, pre-filled by the app. See docs/data/B1_RECORDING_GUIDE.md, "Correcting".
# 1. transcript: change it to exactly what you said, fillers ("um", «امم») and repeats included.
#    Keep whisper_text as it is.
# 2. speech_start_s / speech_end_s: the first and last speech you hear, in seconds.
# 3. removed_at_end: for each word, real: true if you really said it, false if not.
# 4. Pauses are in the .labels.txt file next to this one (open it in Audacity).
# 5. When both files are done, set status: corrected.
"""


def dump_annotation(a: B1Annotation) -> str:
    body = yaml.dump(
        a.model_dump(mode="json"), Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100
    )
    return ANNOTATION_HEADER + body


def load_annotation(content: str) -> B1Annotation:
    return B1Annotation.model_validate(yaml.safe_load(content))


def prefill(report: SpeechReport, stem: str, meta: Prefill) -> tuple[B1Annotation, str]:
    """The report as an annotation to correct, plus its Audacity label track."""
    language, number = parse_stem(stem)
    clip_type, code_switching, long_sentences = CLIP_PLAN[number]
    regions = report.speech_regions
    start = regions[0].start_s if regions else 0.0
    end = regions[-1].end_s if regions else max(report.transcript.duration_s, start + 0.01)
    annotation = B1Annotation(
        clip=report.source_name,
        language=language,
        number=number,
        clip_type=clip_type,
        code_switching=code_switching and language == "ar",
        long_sentences=long_sentences and language == "ar",
        prefill=meta,
        speech_start_s=round(start, 3),
        speech_end_s=round(max(end, start + 0.01), 3),
        whisper_text=report.transcript.text,
        transcript=report.transcript.text,
        removed_at_end=tuple(
            RemovedWord(text=w.text, start_s=w.start_s, end_s=w.end_s)
            for w in report.words_outside_speech
        ),
    )
    labels = write_labels(
        [
            LabelledPause(
                start_s=p.start_s, end_s=p.end_s, label=_LABEL_OF.get(p.position, UNDECIDED)
            )
            for p in report.pauses
        ]
    )
    return annotation, labels


# --- scoring ----------------------------------------------------------------------------------


def words(text: str, language: SpeechLanguage) -> list[str]:
    """Normalised spoken words; punctuation-only tokens dropped."""
    return [t for t in (normalize_token(x, language) for x in text.split()) if t]


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: Sequence[str], hyp: Sequence[str]) -> float | None:
    return edit_distance(ref, hyp) / len(ref) if ref else None


def words_per_sentence_mark(report: SpeechReport, cfg: PauseSettings) -> float | None:
    ws = report.transcript.words
    marks = sum(1 for w in ws if pauses.position_after(w.text, cfg) == "sentence_boundary")
    return len(ws) / marks if marks else None


def _value(report: SpeechReport, key: str) -> float | None:
    m = report.measurements.get(key)
    return None if m is None else m.value


def score_clip(
    a: B1Annotation,
    labelled: Sequence[LabelledPause],
    report: SpeechReport,
    ev: EvaluationSettings,
    pause_cfg: PauseSettings,
    *,
    pileup_ratio_max: float | None,
) -> B1ClipScore:
    if a.status != "corrected":
        raise ValueError(f"{a.clip}: not corrected yet (status: {a.status})")
    if any(p.label == UNDECIDED for p in labelled):
        raise ValueError(f"{a.clip}: a pause label is still '?'")
    lexicon = fillers.load_lexicon(a.language)
    ref = words(a.transcript, a.language)
    hyp = words(report.transcript.text, a.language)
    ref_nh = [w for w in ref if not lexicon.is_hesitation(w)]
    hyp_nh = [w for w in hyp if not lexicon.is_hesitation(w)]
    wpm_true = len(ref) / (a.speech_end_s - a.speech_start_s) * 60
    wpm_measured = _value(report, "rate.wpm")
    err = 100.0 if wpm_measured is None else abs(wpm_measured - wpm_true) / wpm_true * 100

    cfg = ev.e2
    true = [
        TruePause(start_s=p.start_s, end_s=p.end_s, position=LABELS[p.label], source="owner")
        for p in labelled
        if p.end_s - p.start_s >= cfg.min_pause_s
    ]
    det = [d for d in report.pauses if d.duration_s >= cfg.min_pause_s]
    matches = e2.match(true, det, cfg)
    scores = [e2.threshold_score(true, det, matches, t, cfg.f1_exclusion_s) for t in cfg.thresholds]
    check = report.language_check
    profile = report.loudness_profile
    return B1ClipScore(
        clip=a.clip,
        language=a.language,
        clip_type=a.clip_type,
        code_switching=a.code_switching,
        duration_s=report.transcript.duration_s,
        words_true=len(ref),
        words_true_no_hesitations=len(ref_nh),
        words_measured=len(hyp),
        wpm_true=wpm_true,
        wpm_measured=wpm_measured,
        wpm_error_pct=err,
        wer=wer(ref, hyp),
        wer_no_hesitations=wer(ref_nh, hyp_nh),
        true_pauses=len(true),
        detected_pauses=len(det),
        tp={str(s.threshold_s): s.tp for s in scores},  # "0.5", "1.0" as in evaluation.yaml
        fp={str(s.threshold_s): s.fp for s in scores},
        fn={str(s.threshold_s): s.fn for s in scores},
        duration_abs_errors_s=tuple(abs(m.duration_error_s) for m in matches),
        positions=tuple(
            (true[m.true_index].position.value, det[m.detected_index].position.value)
            for m in matches
        ),
        detected_positions=tuple(d.position.value for d in det),
        words_per_sentence_mark=words_per_sentence_mark(report, pause_cfg),
        removed_at_end=len(a.removed_at_end),
        removed_real=sum(1 for w in a.removed_at_end if w.real),
        clipping_share=_value(report, "loudness.clipping_share"),
        possible_clipping_share=_value(report, "loudness.possible_clipping_share"),
        pileup_ratio_max=pileup_ratio_max,
        snr_db=None if profile is None else profile.snr_db,
        detected_language=None if check is None else check.detected,
        language_probability=None if check is None else check.probability,
        language_mismatch=None if check is None else check.mismatch,
    )


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarise(scores: Sequence[B1ClipScore]) -> dict[str, Any]:
    """Descriptive summary for one language (E2 Arabic is descriptive only, FINAL_10 §2)."""
    keys = sorted({k for s in scores for k in s.tp})
    pairs = [p for s in scores for p in s.positions]
    mid_as_boundary, boundary_as_mid = e2.boundary_mid_errors(pairs)
    detected = [x for s in scores for x in s.detected_positions]
    errors = [x for s in scores for x in s.duration_abs_errors_s]
    ratios = [s.pileup_ratio_max for s in scores if s.pileup_ratio_max is not None]
    return {
        "clips": len(scores),
        "wpm_error_pct_mean": _mean([s.wpm_error_pct for s in scores]),
        "wer_mean": _mean([s.wer for s in scores if s.wer is not None]),
        "wer_no_hesitations_mean": _mean(
            [s.wer_no_hesitations for s in scores if s.wer_no_hesitations is not None]
        ),
        "pause_f1": {
            k: f1(
                sum(s.tp[k] for s in scores),
                sum(s.fp[k] for s in scores),
                sum(s.fn[k] for s in scores),
            )
            for k in keys
        },
        "duration_error_s_mean": _mean(errors),
        "position_table": e2.position_table(pairs),
        "mid_as_boundary": mid_as_boundary,
        "boundary_as_mid": boundary_as_mid,
        "unknown_share": (
            sum(1 for x in detected if x == PausePosition.UNKNOWN.value) / len(detected)
            if detected
            else None
        ),
        "removed_at_end": sum(s.removed_at_end for s in scores),
        "removed_real": sum(s.removed_real for s in scores),
        "clips_with_clipping": sum(1 for s in scores if (s.clipping_share or 0) > 0),
        "clips_with_possible_clipping": sum(
            1 for s in scores if (s.possible_clipping_share or 0) > 0
        ),
        "pileup_ratio_max": max(ratios) if ratios else None,
        "language_mismatches": sum(1 for s in scores if s.language_mismatch),
        "language_probability_min": min(
            (s.language_probability for s in scores if s.language_probability is not None),
            default=None,
        ),
    }
