"""synth/split.py: C1 split by script, stratified by voice, with a count check (ADR 0011, 0014)."""

from typing import Any

import pytest

from presenceai.evalconfig import load_evaluation
from presenceai.schemas.c1 import ClipTruth
from presenceai.synth import scripts, split

EV = load_evaluation()
TEXT = "Good morning, everyone. Today we plan."


def all_scripts() -> tuple[Any, ...]:
    return tuple(
        scripts.parse_script(f"s{v * 10 + k + 1:02d}", voice, "T", TEXT)
        for v, voice in enumerate(EV.c1.voices)
        for k in range(EV.c1.scripts_per_voice)
    )


def truth(script_id: str, voice: str, render: int, wpm: float) -> ClipTruth:
    return ClipTruth(
        clip_id=f"{script_id}_r{render}", script_id=script_id, voice=voice, render=render,
        sample_rate=24_000, duration_s=40.0, words=110, fillers=0, speech_onset_s=0.5,
        speech_offset_s=39.5, speech_spans=((0.5, 39.5),), wpm=wpm, articulation_wpm=wpm + 10,
        target_wpm=wpm, speed=1.0, attempts=1, pauses=(), internal_silences=(), text=TEXT,
    )  # fmt: skip


def truths(scr: tuple[Any, ...]) -> list[ClipTruth]:
    return [truth(s.id, s.voice, r, w) for s in scr for r, w in enumerate(EV.c1.target_wpm)]


def test_three_of_ten_scripts_per_voice_go_to_development():
    parts = split.split_scripts(all_scripts(), EV.e2.split)
    for voice in EV.c1.voices:
        ids = [s.id for s in all_scripts() if s.voice == voice]
        assert sum(parts[i] == "dev" for i in ids) == 3


def test_split_is_reproducible_and_seed_dependent():
    a = split.split_scripts(all_scripts(), EV.e2.split)
    assert a == split.split_scripts(all_scripts(), EV.e2.split)
    other = EV.e2.split.model_copy(update={"seed": 1})
    assert a != split.split_scripts(all_scripts(), other)


def test_all_clips_of_a_script_share_a_part():
    scr = all_scripts()
    parts = split.split_scripts(scr, EV.e2.split)
    table = split.assign(truths(scr), parts)
    for t, part in table:
        assert part == parts[t.script_id]


def test_counts_cover_every_band_and_voice_in_both_parts():
    scr = all_scripts()
    parts = split.split_scripts(scr, EV.e2.split)
    counts = split.counts(split.assign(truths(scr), parts), EV.c1)
    split.check_counts(counts, EV.c1)  # no error
    text = split.format_counts(counts)
    assert "dev" in text and "test" in text and "110-140" in text and "bm_george" in text
    assert sum(v for (p, kind, _), v in counts.items() if kind == "voice") == 300


def test_an_empty_band_stops_with_an_error():
    scr = all_scripts()
    parts = split.split_scripts(scr, EV.e2.split)
    slow = [t.model_copy(update={"wpm": 120.0}) for t in truths(scr)]  # nothing above 140
    with pytest.raises(ValueError, match="170-200"):
        split.check_counts(split.counts(split.assign(slow, parts), EV.c1), EV.c1)
