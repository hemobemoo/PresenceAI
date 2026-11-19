"""experiments/e2_rate_pauses/word_side_dev.py: the overlap variant of find_pauses (C1 dev only,
ADR 0016 open item 6). No models, no audio."""

import importlib.util
import sys
from typing import Any

from presenceai.evalconfig import REPO_ROOT
from presenceai.schemas.speech import PausePosition, SpeechRegion
from presenceai.speech import pauses
from tests.unit.test_speech_pauses import _transcript

spec = importlib.util.spec_from_file_location(
    "word_side_dev", REPO_ROOT / "experiments" / "e2_rate_pauses" / "word_side_dev.py"
)
assert spec is not None and spec.loader is not None
dev = importlib.util.module_from_spec(spec)
sys.modules["word_side_dev"] = dev
spec.loader.exec_module(dev)

TOKENS = [("One.", 0.0, 0.5), ("Two,", 1.1, 2.1), ("three", 2.6, 3.0), ("four.", 3.1, 3.4)]
REGIONS = (SpeechRegion(start_s=0.0, end_s=1.5), SpeechRegion(start_s=2.5, end_s=3.4))


def test_overlap_variant_puts_the_stretched_word_before_the_pause(speech_cfg: Any):
    moved = [0]
    find = dev.overlap_rule(0.10, "skip", moved)
    (event,) = find(REGIONS, _transcript(TOKENS), speech_cfg)
    # "Two," lies 0.4 s outside the pause before it and 0 s after it: before the pause
    assert event.word_before == 1 and event.position == PausePosition.CLAUSE_BOUNDARY
    assert moved == [1]
    (app,) = pauses.find_pauses(REGIONS, _transcript(TOKENS), speech_cfg)
    assert app.word_before == 0  # the app's rule is unchanged


def test_overlap_variant_unplaced_word(speech_cfg: Any):
    tokens = [("One.", 0.0, 0.5), ("Two,", 1.4, 2.6), ("three", 2.7, 3.0), ("four.", 3.1, 3.4)]
    (skip,) = dev.overlap_rule(0.10, "skip", [0])(REGIONS, _transcript(tokens), speech_cfg)
    (unk,) = dev.overlap_rule(0.10, "unknown", [0])(REGIONS, _transcript(tokens), speech_cfg)
    assert skip.word_before == 0 and skip.position == PausePosition.SENTENCE_BOUNDARY
    assert unk.position == PausePosition.UNKNOWN
