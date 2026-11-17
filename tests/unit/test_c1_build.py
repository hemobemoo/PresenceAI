"""synth/c1.py: C1 clips from separately generated phrases (ADR 0014 §3). Fake TTS only."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from fake_tts import FakeTTS, edges

from presenceai.evalconfig import load_evaluation
from presenceai.schemas.c1 import ClipPlan, GapPlan
from presenceai.synth import c1, scripts

CFG = load_evaluation().c1
SR = 24_000
TEXT = (
    "Good morning, everyone. Today I want to share | a simple idea about planning. "
    "Small steps help us | finish big tasks."
)


def script(text: str = TEXT, voice: str = "af_heart") -> Any:
    return scripts.parse_script("s01", voice, "Planning", text)


# --- trimming (owner's point 7) -------------------------------------------------------------


@pytest.mark.parametrize("text", ["sun", "fish", "help", "his", "half", "say so"])
def test_quiet_onsets_and_endings_are_not_cut(text: str):
    """s, f and h edges lie 35-45 dB below the vowel. The 55 dB threshold keeps them: the
    trimmed edges are within 2 ms of where the sound really starts and ends."""
    x = FakeTTS().synth(text, "v", 1.0)
    bounds = c1.trim(x, SR, CFG.trim)
    assert bounds is not None
    a, b = bounds
    true_a, true_b = edges(text)
    assert abs(a / SR - true_a) < 0.002
    assert abs(b / SR - true_b) < 0.002


def test_a_higher_threshold_would_cut_the_quiet_h():
    """Control: at 40 dB below the peak, the -45 dB "h" onset is lost."""
    strict = CFG.trim.model_copy(update={"threshold_db_below_peak": 40.0})
    bounds = c1.trim(FakeTTS().synth("help", "v", 1.0), SR, strict)
    assert bounds is not None
    assert bounds[0] / SR - edges("help")[0] > 0.03


def test_threshold_is_recorded_in_the_config():
    assert CFG.trim.threshold_db_below_peak == 55
    assert CFG.trim.frame_ms == 5


def test_silent_phrase_gives_no_bounds():
    assert c1.trim(np.zeros(SR, np.float32), SR, CFG.trim) is None


def test_kokoro_like_internal_silence_is_found():
    tts = FakeTTS(internal_silence={"share": 0.4})
    x = tts.synth("we share the plan", "v", 1.0)
    found = c1.internal_silences(x, SR, CFG.trim)
    assert len(found) == 1
    a, b = found[0]
    assert (b - a) / SR == pytest.approx(0.4, abs=0.011)


def test_short_word_gaps_are_not_internal_silences():
    x = FakeTTS().synth("we share the plan today", "v", 1.0)  # 20 ms gaps
    assert c1.internal_silences(x, SR, CFG.trim) == []


# --- planning ---------------------------------------------------------------------------------


def test_plan_is_reproducible_and_in_range():
    s = script()
    p1 = c1.plan_clip(s, 2, 155.0, c1.clip_rng(CFG, 0, 2), CFG)
    p2 = c1.plan_clip(s, 2, 155.0, c1.clip_rng(CFG, 0, 2), CFG)
    assert p1 == p2
    assert len(p1.gaps) == len(s.phrases) - 1
    lo_s, hi_s = CFG.gaps.short_s
    lo_p, hi_p = CFG.gaps.pause_s
    for g in p1.gaps:
        assert lo_s <= g.seconds <= hi_s or lo_p <= g.seconds <= hi_p
        assert g.boundary == s.phrases[g.after_phrase].boundary_after


def test_about_a_third_of_clips_get_one_or_two_fillers():
    s = script()
    plans = [
        c1.plan_clip(s, r, 155.0, c1.clip_rng(CFG, k, r), CFG) for k in range(60) for r in range(5)
    ]
    with_fillers = [p for p in plans if any(g.filler for g in p.gaps)]
    assert 0.25 < len(with_fillers) / len(plans) < 0.41
    for p in with_fillers:
        n = sum(1 for g in p.gaps if g.filler)
        assert 1 <= n <= 2
        assert all(g.filler in CFG.fillers.tokens for g in p.gaps if g.filler)


def test_pause_probability_follows_the_boundary_type():
    s = script()
    gaps = [
        g for k in range(200) for g in c1.plan_clip(s, 0, 155.0, c1.clip_rng(CFG, k, 0), CFG).gaps
    ]
    for kind, p in CFG.gaps.pause_probability.items():
        sel = [g for g in gaps if g.boundary == kind]
        share = sum(g.seconds >= CFG.gaps.pause_s[0] for g in sel) / len(sel)
        assert share == pytest.approx(p, abs=0.08), kind


# --- assembling: truth comes from the construction ------------------------------------------


def fixed_plan(s: Any, gaps: Sequence[tuple[float, str | None]]) -> ClipPlan:
    return ClipPlan(
        clip_id=f"{s.id}_r0",
        script_id=s.id,
        voice=s.voice,
        render=0,
        target_wpm=155.0,
        gaps=tuple(
            GapPlan(
                after_phrase=i,
                boundary=s.phrases[i].boundary_after,
                seconds=sec,
                filler=fill,
                after_filler_s=0.1 if fill else None,
            )
            for i, (sec, fill) in enumerate(gaps)
        ),
    )


def test_true_pauses_are_exactly_the_inserted_silences():
    s = script()  # phrases: "Good morning," | "everyone." | "Today I want to share" | ...
    gaps = [(0.8, None), (1.5, None), (0.1, None), (0.6, None), (0.1, None)]
    plan = fixed_plan(s, gaps[: len(s.phrases) - 1])
    clean, truth, _ = c1.assemble(s, plan, FakeTTS(), 1.0, CFG)
    assert [round(p.duration_s, 6) for p in truth.pauses] == [0.8, 1.5, 0.6]
    assert [p.position for p in truth.pauses] == [
        "clause_boundary",
        "sentence_boundary",
        "sentence_boundary",
    ]
    assert all(p.source == "inserted" for p in truth.pauses)
    # the 0.1 s gap after "share |" is not a pause
    lead = CFG.edge_silence_s
    assert truth.speech_onset_s == pytest.approx(lead)
    assert clean.size / SR == pytest.approx(truth.duration_s)


def test_true_wpm_is_measured_on_the_finished_audio():
    s = script()
    plan = fixed_plan(s, [(0.5, None)] * (len(s.phrases) - 1))
    clean, truth, _ = c1.assemble(s, plan, FakeTTS(), 1.25, CFG)
    bounds = c1.trim(clean, SR, CFG.trim)  # measured again on the whole clip
    assert bounds is not None
    a, b = bounds
    assert truth.speech_onset_s == pytest.approx(a / SR, abs=0.005)
    assert truth.speech_offset_s == pytest.approx(b / SR, abs=0.005)
    assert truth.words == s.words
    assert truth.wpm == pytest.approx(
        truth.words * 60 / (truth.speech_offset_s - truth.speech_onset_s)
    )
    assert truth.articulation_wpm > truth.wpm


def test_fillers_are_spoken_counted_and_followed_by_a_short_gap():
    s = script()
    gaps: list[tuple[float, str | None]] = [(0.7, "um")]
    gaps += [(0.1, None)] * (len(s.phrases) - 2)
    tts = FakeTTS()
    _, truth, _ = c1.assemble(s, fixed_plan(s, gaps), tts, 1.0, CFG)
    assert any(text == "um" for text, _, _ in tts.calls)
    assert truth.fillers == 1 and truth.words == s.words + 1
    assert [round(p.duration_s, 6) for p in truth.pauses] == [0.7]  # 0.1 s after "um" is no pause
    assert truth.pauses[0].position == "clause_boundary"  # where the filler was inserted


def test_kokoro_pauses_join_the_truth_from_025_s():
    s = script("Today we share the plan | together. We review it slowly.")
    tts = FakeTTS(internal_silence={"share": 0.35, "review": 0.15})
    _, truth, stats = c1.assemble(s, fixed_plan(s, [(0.1, None), (0.1, None)]), tts, 1.0, CFG)
    kokoro = [p for p in truth.pauses if p.source == "kokoro"]
    assert len(kokoro) == 1 and kokoro[0].duration_s == pytest.approx(0.35, abs=0.011)
    assert kokoro[0].position == "mid_sentence"
    assert len(truth.internal_silences) == 2  # both reported, only one is a pause
    assert stats.phrases == len(s.phrases)


def test_noise_floor_level():
    rng = np.random.default_rng(1)
    x = c1.add_noise_floor(np.zeros(SR * 5, np.float32), -60.0, rng)
    rms_db = 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    assert rms_db == pytest.approx(-60.0, abs=0.2)


@pytest.mark.parametrize("snr", [20.0, 10.0])
def test_noise_is_mixed_at_the_requested_snr(snr: float):
    s = script()
    clean, truth, _ = c1.assemble(
        s, fixed_plan(s, [(0.5, None)] * (len(s.phrases) - 1)), FakeTTS(), 1.0, CFG
    )
    noisy = c1.mix_noise(clean, truth.speech_spans, SR, snr, np.random.default_rng(3), "pink")
    noise = noisy - clean
    mask = c1.span_mask(truth.speech_spans, SR, clean.size)
    p_s = np.mean(clean[mask].astype(np.float64) ** 2)
    p_n = np.mean(noise.astype(np.float64) ** 2)
    assert 10 * np.log10(p_s / p_n) == pytest.approx(snr, abs=0.05)


def test_pink_noise_falls_3_db_per_octave():
    x = c1.pink_noise(2**18, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)

    def band(lo: float) -> float:
        return float(np.mean(spec[(f >= lo) & (f < 2 * lo)]))

    assert 10 * np.log10(band(500) / band(1000)) == pytest.approx(3.0, abs=0.5)


# --- rendering to a target rate -----------------------------------------------------------


LONG = " ".join([TEXT] * 5)  # 110 words, like a real script
WIDE = CFG.model_copy(update={"clip_s": (1.0, 500.0)})


def test_render_reaches_the_target_rate_by_correcting_speed():
    s = script(LONG)
    plan = c1.plan_clip(s, 0, 137.0, c1.clip_rng(CFG, 0, 0), CFG)
    result = c1.render(s, plan, FakeTTS(), WIDE, wps_at_speed_1=1.0)  # a bad first guess
    assert result.truth is not None
    assert abs(result.truth.wpm - 137.0) <= CFG.accept_within_wpm
    assert result.truth.attempts >= 2


def test_render_drops_a_clip_it_cannot_bring_into_range():
    s = script(LONG)
    plan = c1.plan_clip(s, 0, 191.0, c1.clip_rng(CFG, 0, 0), CFG)
    slow_cfg = WIDE.model_copy(update={"speed_range": (0.5, 0.55)})  # cannot speak fast enough
    result = c1.render(s, plan, FakeTTS(), slow_cfg, wps_at_speed_1=3.0)
    assert result.truth is None and "wpm" in (result.dropped or "")
    assert result.attempts == CFG.max_rerenders + 1


def test_render_drops_a_clip_outside_the_length_range():
    s = script()  # 22 words: far shorter than 35 s
    plan = c1.plan_clip(s, 0, 155.0, c1.clip_rng(CFG, 0, 0), CFG)
    result = c1.render(s, plan, FakeTTS(), CFG, wps_at_speed_1=3.1)
    assert result.truth is None and "clip is" in (result.dropped or "")


def test_rate_band_uses_the_measured_value():
    assert c1.rate_band(112.0, CFG.rate_bands) == "110-140"
    assert c1.rate_band(170.0, CFG.rate_bands) == "170-200"
    assert c1.rate_band(205.0, CFG.rate_bands) is None
