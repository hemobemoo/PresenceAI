"""speech/fillers.py: lexicon fillers, discourse markers, acoustic candidates, fusion (ADR 0012)."""

from typing import Any

import numpy as np
import pytest

from presenceai.schemas.assessment import Confidence
from presenceai.schemas.speech import FillerKind
from presenceai.speech import fillers, pitch

# --- lexicon -------------------------------------------------------------------------------


def test_english_hesitations_and_markers_are_separate(sb: Any):
    t = sb.spaced("Um, so I uh think, like, you know, it works.")
    events = fillers.find_lexical(t, fillers.load_lexicon("en"))
    hes = [
        t.span(e.first_word, e.last_word + 1).text
        for e in events
        if e.kind == FillerKind.HESITATION
    ]
    mk = [
        t.span(e.first_word, e.last_word + 1).text
        for e in events
        if e.kind == FillerKind.DISCOURSE_MARKER
    ]
    assert hes == ["Um,", "uh"]
    assert mk == ["so", "like,", "you know,"]


@pytest.mark.parametrize("token", ["Ummm", "UH.", "Hmm,", "erm", "Mm"])
def test_spelling_variants_of_hesitations(sb: Any, token: str):
    t = sb.spaced(f"well {token} yes")
    (e,) = fillers.find_lexical(t, fillers.load_lexicon("en"))
    assert e.kind == FillerKind.HESITATION and e.first_word == e.last_word == 1


@pytest.mark.parametrize("text", ["umbrella likely", "I meant it", "summer"])
def test_words_that_only_look_like_fillers_are_not_counted(sb: Any, text: str):
    assert fillers.find_lexical(sb.spaced(text), fillers.load_lexicon("en")) == ()


def test_arabic_hesitations_and_markers_after_normalisation(sb: Any):
    # يعني and طيب are discourse markers (ADR 0012 changes FINAL_04 §5), امم and آه are fillers
    t = sb.spaced("امم يَعْنِي الفكرة آه طيـــب", language="ar")
    events = fillers.find_lexical(t, fillers.load_lexicon("ar"))
    kinds = [(e.first_word, e.kind) for e in events]
    assert kinds == [
        (0, FillerKind.HESITATION),
        (1, FillerKind.DISCOURSE_MARKER),
        (3, FillerKind.HESITATION),
        (4, FillerKind.DISCOURSE_MARKER),
    ]


# --- acoustic candidates -------------------------------------------------------------------

ACOUSTIC_WORDS = [("Today", 0.0, 0.4), ("we", 1.0, 1.2)]  # a 0.6 s gap with no word


def _acoustic(sb: Any, cfg: Any, segments: Any, words: Any = ACOUSTIC_WORDS, lexical: Any = ()):
    t = sb.transcript(words)
    track = sb.pitch_track(1.5, segments=segments)
    return fillers.acoustic_candidates((sb.region(0.0, 1.2),), t, track, cfg, lexical)


def test_steady_voiced_sound_without_a_word_is_a_candidate(sb: Any, speech_cfg: Any):
    (c,) = _acoustic(sb, speech_cfg, [(0.5, 0.8, 120.0, 120.0)])
    assert c.start_s == pytest.approx(0.5, abs=0.02) and c.end_s == pytest.approx(0.8, abs=0.02)
    assert c.pitch_sd_semitones < 0.1 and c.voiced_fraction > 0.9
    assert c.agrees_with_lexicon is False


def test_gliding_pitch_is_not_a_filler(sb: Any, speech_cfg: Any):
    assert _acoustic(sb, speech_cfg, [(0.5, 0.8, 100.0, 160.0)]) == ()  # ~8 semitones


def test_unvoiced_gap_is_not_a_filler(sb: Any, speech_cfg: Any):
    assert _acoustic(sb, speech_cfg, []) == ()


def test_too_short_is_not_a_filler(sb: Any, speech_cfg: Any):
    assert _acoustic(sb, speech_cfg, [(0.5, 0.6, 120.0, 120.0)]) == ()  # 0.10 s < 0.15 s


def test_outside_speech_regions_is_ignored(sb: Any, speech_cfg: Any):
    t = sb.transcript(ACOUSTIC_WORDS)
    track = sb.pitch_track(1.5, segments=[(0.5, 0.8, 120.0, 120.0)])
    regions = (sb.region(0.0, 0.4), sb.region(1.0, 1.2))  # the VAD saw no speech in the gap
    assert fillers.acoustic_candidates(regions, t, track, speech_cfg, ()) == ()


def test_filler_right_next_to_a_word_depends_on_word_timing(sb: Any, speech_cfg: Any):
    """Owner's check (2026-11-17): a hesitation directly after a word. If Whisper's word end is
    accurate the detector sees it; if Whisper stretches the word over the hesitation, the
    detector cannot. This test pins down that limitation (LIMITATIONS.md; measured in E3)."""
    hesitation = [(0.4, 0.7, 120.0, 120.0)]  # starts exactly where "Today" ends
    accurate = [("Today", 0.0, 0.4), ("we", 1.0, 1.2)]
    stretched = [("Today", 0.0, 0.72), ("we", 1.0, 1.2)]
    assert len(_acoustic(sb, speech_cfg, hesitation, words=accurate)) == 1
    assert _acoustic(sb, speech_cfg, hesitation, words=stretched) == ()


def test_candidate_overlapping_a_lexicon_filler_agrees(sb: Any, speech_cfg: Any):
    words = [("Today", 0.0, 0.4), ("uh", 0.5, 0.8), ("we", 1.0, 1.2)]
    t = sb.transcript(words)
    lexical = fillers.find_lexical(t, fillers.load_lexicon("en"))
    # the hesitation has a word, so the acoustic detector looks only where there is no word
    track = sb.pitch_track(1.5, segments=[(0.5, 0.8, 120.0, 120.0)])
    assert fillers.acoustic_candidates((sb.region(0.0, 1.2),), t, track, speech_cfg, lexical) == ()
    # with Whisper's "uh" slightly short, the uncovered tail is a candidate that agrees
    words2 = [("Today", 0.0, 0.4), ("uh", 0.5, 0.6), ("we", 1.0, 1.2)]
    t2 = sb.transcript(words2)
    lex2 = fillers.find_lexical(t2, fillers.load_lexicon("en"))
    track2 = sb.pitch_track(1.5, segments=[(0.5, 0.85, 120.0, 120.0)])
    (c,) = fillers.acoustic_candidates((sb.region(0.0, 1.2),), t2, track2, speech_cfg, lex2)
    assert c.agrees_with_lexicon is True


def test_real_pyin_finds_a_steady_tone_between_words(sb: Any, speech_cfg: Any):
    sr = 16_000
    x = np.zeros(int(1.5 * sr), dtype=np.float32)
    t = np.arange(int(0.3 * sr)) / sr
    x[int(0.5 * sr) : int(0.8 * sr)] = 0.3 * np.sin(2 * np.pi * 150 * t)
    track = pitch.track(x, sr, speech_cfg.pitch)
    tr = sb.transcript(ACOUSTIC_WORDS)
    (c,) = fillers.acoustic_candidates((sb.region(0.0, 1.2),), tr, track, speech_cfg, ())
    assert c.start_s == pytest.approx(0.5, abs=0.06) and c.end_s == pytest.approx(0.8, abs=0.06)


# --- measures --------------------------------------------------------------------------------


def _measure(
    sb: Any,
    cfg: Any,
    text: str,
    *,
    language: str = "en",
    verbatim: bool = False,
    p: float = 0.95,
    acoustic: Any = (),
):
    t = sb.spaced(text, step=0.5, language=language, p=p)
    regions = (sb.region(0.0, t.words[-1].end_s if t.words else 0.0),)
    lex = fillers.find_lexical(t, fillers.load_lexicon(language))  # pyright: ignore[reportArgumentType]
    return fillers.summarise(lex, acoustic, t, regions, cfg, verbatim=verbatim)


LONG = " ".join(["word"] * 76 + ["um", "so", "uh", "like"])  # 80 tokens: 2 fillers, 2 markers


def test_rates_use_non_filler_words_and_are_lower_bounds(sb: Any, speech_cfg: Any):
    m, nm = _measure(sb, speech_cfg, LONG)
    assert nm == {}
    assert m["fillers.count"].value == 2
    assert m["fillers.per_100_words"].value == pytest.approx(2 / 78 * 100)
    assert "lower bound" in m["fillers.per_100_words"].method
    assert m["discourse_markers.count"].value == 2
    assert "upper bound" in m["discourse_markers.per_100_words"].method
    assert m["discourse_markers.per_100_words"].confidence == Confidence.LOW
    assert [e.transcript.text for e in m["fillers.count"].evidence if e.transcript] == ["um", "uh"]


def test_english_confidence_is_capped_at_medium_without_verbatim_model(sb: Any, speech_cfg: Any):
    m, _ = _measure(sb, speech_cfg, LONG)  # 40 s, p = 0.95 would be "high"
    assert m["fillers.per_100_words"].confidence == Confidence.MEDIUM
    m_v, _ = _measure(sb, speech_cfg, LONG, verbatim=True)
    assert m_v["fillers.per_100_words"].confidence == Confidence.HIGH
    assert "lower bound" not in m_v["fillers.per_100_words"].method


def test_arabic_confidence_is_capped_at_low(sb: Any, speech_cfg: Any):
    text = " ".join(["كلمة"] * 78 + ["امم", "اه"])
    m, _ = _measure(sb, speech_cfg, text, language="ar")
    assert m["fillers.per_100_words"].confidence == Confidence.LOW


def test_acoustic_only_candidates_widen_the_range(sb: Any, speech_cfg: Any):
    from presenceai.schemas.speech import AcousticCandidate

    cands = (
        AcousticCandidate(
            start_s=50.0,
            end_s=50.3,
            voiced_fraction=1.0,
            pitch_sd_semitones=0.2,
            agrees_with_lexicon=False,
        ),
        AcousticCandidate(
            start_s=60.0,
            end_s=60.3,
            voiced_fraction=1.0,
            pitch_sd_semitones=0.2,
            agrees_with_lexicon=True,
        ),
    )
    m, _ = _measure(sb, speech_cfg, LONG, acoustic=cands)
    assert m["fillers.acoustic_candidates"].value == 2
    assert m["fillers.per_100_words_upper"].value == pytest.approx((2 + 1) / 78 * 100)
    assert m["fillers.per_100_words_upper"].confidence == Confidence.LOW


def test_no_words_is_not_measurable(sb: Any, speech_cfg: Any):
    t = sb.transcript([])
    m, nm = fillers.summarise((), (), t, (), speech_cfg, verbatim=False)
    assert m == {} and "fillers.per_100_words" in nm
