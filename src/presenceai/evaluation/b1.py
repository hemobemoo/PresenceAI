"""B1: the owner's annotated recordings as the real-speech check (ADR 0016).

- `prefill` turns an app report into the two annotation files: `<clip>.yaml` (transcript and
  speech start/end, corrected by the owner) and `<clip>.labels.txt` (the detected pauses, all
  labelled `?`);
- `label_pauses` fills the labels from the punctuation of the owner's corrected transcript,
  aligned to Whisper's timed words. Pause times stay as the app detected them, so B1 checks
  pause *positions* (ADR 0014) but not pause detection or duration (ADR 0016 amendment);
- `score_clip` compares a corrected annotation with a report, using the E2 matching rules
  unchanged (`evaluation.e2`, ADR 0014 §6). B1 is descriptive: no pass/fail, no tuning target;
- clip names and the clip plan follow `docs/data/B1_RECORDING_GUIDE.md`.

Results hold numbers only; transcript text never leaves data/b1/.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
    """One `start<TAB>end<TAB>label` line per pause."""
    return "".join(f"{p.start_s:.6f}\t{p.end_s:.6f}\t{p.label}\n" for p in items)


def read_labels(content: str) -> tuple[LabelledPause, ...]:
    out: list[LabelledPause] = []
    for n, line in enumerate(content.splitlines(), start=1):
        if not line.strip() or line.startswith("\\"):  # frequency lines of other label tools
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
#    Punctuate by meaning: . ? ! ؟ where a sentence ends, , ، ; ؛ where a clause ends,
#    not where you paused.
# 4. Run `run.py label <clip>`: it fills the .labels.txt next to this file from your
#    punctuation. Decide any "?" by ear with `run.py listen <clip> <pause-number>`.
#    Never change pause times.
# 5. Set status: corrected. Comments you add here are not kept; use notes: instead.
"""


def dump_annotation(a: B1Annotation) -> str:
    body = yaml.dump(
        a.model_dump(mode="json"), Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100
    )
    return ANNOTATION_HEADER + body


def load_annotation(content: str) -> B1Annotation:
    return B1Annotation.model_validate(yaml.safe_load(content))


def prefill(report: SpeechReport, stem: str, meta: Prefill) -> tuple[B1Annotation, str]:
    """The report as an annotation to correct, plus its labels file (every label "?")."""
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
        [LabelledPause(start_s=p.start_s, end_s=p.end_s, label=UNDECIDED) for p in report.pauses]
    )
    return annotation, labels


# --- pause labels from the owner's transcript (ADR 0016 amendment) ---------------------------

# The owner's rule (2026-11-19): punctuation after the last word before the pause.
SENTENCE_MARKS = frozenset(".?!؟")
CLAUSE_MARKS = frozenset(",،;؛")
_CLOSERS = "\"'”’»)]}"


@dataclass(frozen=True)
class Token:
    raw: str  # as typed, punctuation included
    norm: str  # normalised for alignment
    start_s: float | None = None  # from the aligned Whisper word; None for added words
    end_s: float | None = None

    @property
    def mid_s(self) -> float | None:
        if self.start_s is None or self.end_s is None:
            return None
        return (self.start_s + self.end_s) / 2


@dataclass(frozen=True)
class PauseLabel:
    number: int  # 1-based, in the labels file's order
    pause: LabelledPause
    label: str
    reason: str | None = None  # why it is "?"
    before: tuple[str, ...] = ()  # up to 3 of the owner's words before / after
    after: tuple[str, ...] = ()


def tokens(text: str, language: SpeechLanguage) -> list[Token]:
    """Words of a transcript; a punctuation-only token ("word ،") joins the word before it."""
    out: list[Token] = []
    for raw in text.split():
        norm = normalize_token(raw, language)
        if norm:
            out.append(Token(raw=raw, norm=norm))
        elif out:
            out[-1] = Token(raw=out[-1].raw + raw, norm=out[-1].norm)
    return out


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[int | None, int | None]]:
    """Word-level edit-distance alignment: (ref index, hyp index) pairs; None = inserted or
    deleted. Substitutions pair up, so a misheard word keeps its position.

    Among alignments with the fewest edits, the one with the most exact matches wins, so an
    added «امم» is an insertion rather than the start of a chain of substitutions.
    """
    n, m = len(ref), len(hyp)
    # cost = (edits, -exact matches), compared as tuples
    d = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = (i, 0)
    for j in range(m + 1):
        d[0][j] = (j, 0)

    def diag(i: int, j: int) -> tuple[int, int]:
        e, neg = d[i - 1][j - 1]
        same = ref[i - 1] == hyp[j - 1]
        return (e + (not same), neg - same)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            up, left = d[i - 1][j], d[i][j - 1]
            d[i][j] = min(diag(i, j), (up[0] + 1, up[1]), (left[0] + 1, left[1]))
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == diag(i, j):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == (d[i - 1][j][0] + 1, d[i - 1][j][1]):
            pairs.append((i - 1, None))  # a word the owner added (no timing)
            i -= 1
        else:
            pairs.append((None, j - 1))  # a Whisper word the owner deleted
            j -= 1
    return pairs[::-1]


def timed_tokens(text: str, report: SpeechReport, language: SpeechLanguage) -> list[Token]:
    """The owner's words, each with the times of the Whisper word it aligns to."""
    mine = tokens(text, language)
    theirs = report.transcript.words
    hyp = [normalize_token(w.text, language) for w in theirs]
    out = list(mine)
    for i, j in align([t.norm for t in mine], hyp):
        if i is not None and j is not None:
            w = theirs[j]
            out[i] = Token(raw=mine[i].raw, norm=mine[i].norm, start_s=w.start_s, end_s=w.end_s)
    return out


def punctuation_label(raw: str) -> str:
    stripped = raw.rstrip(_CLOSERS)
    mark = stripped[-1:] if stripped else ""
    if mark in SENTENCE_MARKS:
        return "sentence"
    if mark in CLAUSE_MARKS:
        return "clause"
    return "mid"


Side = Literal["before", "after"]


def side_of_pause(t: Token, p: LabelledPause, margin_s: float) -> Side | None:
    """Which side of the pause a timed word is on; None when it cannot be told.

    By midpoint first, as in `pauses.find_pauses`. Whisper often stretches a word into the
    silence next to it (very often in Arabic), so its midpoint can fall inside the pause. Such a
    word goes to the side where more of it lies outside the pause, if that part is longer by
    more than `margin_s`. A word about equally on both sides, or (almost) all inside, is None.
    """
    if t.start_s is None or t.end_s is None:
        raise ValueError("side_of_pause needs a timed word")
    mid = (t.start_s + t.end_s) / 2
    if mid <= p.start_s + 1e-9:
        return "before"
    if mid >= p.end_s - 1e-9:
        return "after"
    outside_before = max(0.0, p.start_s - t.start_s)
    outside_after = max(0.0, t.end_s - p.end_s)
    if outside_after - outside_before > margin_s + 1e-9:
        return "after"
    if outside_before - outside_after > margin_s + 1e-9:
        return "before"
    return None


def label_pauses(
    pauses_in: Sequence[LabelledPause], words_in: Sequence[Token], *, margin_s: float
) -> list[PauseLabel]:
    """Label each pause from the owner's punctuation after the last word before it.

    Each timed word is put on one side of the pause by `side_of_pause`. The label is "?" when
    there is no timed word on one side, or when anything sits between the last word before and
    the first word after: words the owner added (no timing, so they could be on either side) or
    a word whose side cannot be told. Times are never changed.
    """
    timed = [i for i, t in enumerate(words_in) if t.mid_s is not None]
    out: list[PauseLabel] = []
    for number, p in enumerate(pauses_in, start=1):
        sides = {i: side_of_pause(words_in[i], p, margin_s) for i in timed}
        before = max((i for i in timed if sides[i] == "before"), default=None)
        after = min((i for i in timed if sides[i] == "after"), default=None)
        k = 0 if before is None else before + 1
        reason = None
        if before is None or after is None:
            reason = "none of your words is timed on one side of the pause"
        elif after != before + 1:
            inside = [i for i in timed if before < i < after]
            reason = (
                "a word sits inside the pause, about equally on both sides"
                if inside
                else "words you added (no timing) are next to the pause"
            )
        label = UNDECIDED if reason else punctuation_label(words_in[k - 1].raw)
        out.append(
            PauseLabel(
                number=number,
                pause=p,
                label=label,
                reason=reason,
                before=tuple(t.raw for t in words_in[max(0, k - 3) : k]),
                after=tuple(t.raw for t in words_in[k : k + 3]),
            )
        )
    return out


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
    if a.labels_method != "transcript_punctuation":
        raise ValueError(f"{a.clip}: run `run.py label` first (labels come from your transcript)")
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
    edges = a.pause_edges_checked
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
        pause_edges_checked=edges,
        # "0.5", "1.0" as in evaluation.yaml; only with hand-checked pause edges (ADR 0016)
        tp={str(s.threshold_s): s.tp for s in scores} if edges else None,
        fp={str(s.threshold_s): s.fp for s in scores} if edges else None,
        fn={str(s.threshold_s): s.fn for s in scores} if edges else None,
        duration_abs_errors_s=tuple(abs(m.duration_error_s) for m in matches) if edges else None,
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
    checked = [s for s in scores if s.pause_edges_checked]
    keys = sorted({k for s in checked for k in s.tp or {}})
    pairs = [p for s in scores for p in s.positions]
    mid_as_boundary, boundary_as_mid = e2.boundary_mid_errors(pairs)
    detected = [x for s in scores for x in s.detected_positions]
    errors = [x for s in checked for x in s.duration_abs_errors_s or ()]
    ratios = [s.pileup_ratio_max for s in scores if s.pileup_ratio_max is not None]
    return {
        "clips": len(scores),
        "wpm_error_pct_mean": _mean([s.wpm_error_pct for s in scores]),
        "wer_mean": _mean([s.wer for s in scores if s.wer is not None]),
        "wer_no_hesitations_mean": _mean(
            [s.wer_no_hesitations for s in scores if s.wer_no_hesitations is not None]
        ),
        # pause detection and duration only for clips whose pause edges were checked by ear
        "pause_f1": {
            k: f1(
                sum((s.tp or {})[k] for s in checked),
                sum((s.fp or {})[k] for s in checked),
                sum((s.fn or {})[k] for s in checked),
            )
            for k in keys
        },
        "duration_error_s_mean": _mean(errors),
        "pause_edges_not_hand_checked": [s.clip for s in scores if not s.pause_edges_checked],
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
