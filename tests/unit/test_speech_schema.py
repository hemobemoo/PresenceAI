"""Speech data types (schemas/speech.py): words carry timestamps, probability and the exact
character span they occupy in the transcript text, so any finding can quote it as evidence."""

import pytest
from pydantic import ValidationError

from presenceai.schemas.assessment import TranscriptSpan
from presenceai.schemas.speech import AsrQuality, Segment, SpeechReport, Transcript, Word


def _word(text: str, start_char: int, start_s: float = 0.0, end_s: float = 0.5) -> Word:
    return Word(
        text=text,
        start_s=start_s,
        end_s=end_s,
        probability=0.9,
        start_char=start_char,
        end_char=start_char + len(text),
    )


def _transcript() -> Transcript:
    w1, w2 = _word("Hello", 0, 0.0, 0.4), _word("everyone.", 6, 0.45, 1.2)
    seg = Segment(start_s=0.0, end_s=1.2, words=(w1, w2), avg_logprob=-0.2, no_speech_prob=0.01)
    return Transcript(language="en", duration_s=2.0, text="Hello everyone.", segments=(seg,))


def test_word_rejects_end_before_start():
    with pytest.raises(ValidationError, match="end_s"):
        Word(text="a", start_s=1.0, end_s=0.5, probability=0.5, start_char=0, end_char=1)


@pytest.mark.parametrize("p", [-0.01, 1.01])
def test_word_probability_is_a_probability(p: float):
    with pytest.raises(ValidationError):
        Word(text="a", start_s=0, end_s=1, probability=p, start_char=0, end_char=1)


def test_word_text_must_match_its_char_span():
    with pytest.raises(ValidationError, match="char"):
        Word(text="abc", start_s=0, end_s=1, probability=0.5, start_char=0, end_char=2)


def test_transcript_words_and_spans_quote_the_text_exactly():
    t = _transcript()
    assert [w.text for w in t.words] == ["Hello", "everyone."]
    span = t.span(0, 2)
    assert isinstance(span, TranscriptSpan)
    assert span.text == "Hello everyone." == t.text[span.start_char : span.end_char]
    assert t.span(1, 2).text == "everyone."


def test_transcript_rejects_word_offsets_that_do_not_match_the_text():
    seg = Segment(
        start_s=0, end_s=1, words=(_word("Hello", 1),), avg_logprob=-0.1, no_speech_prob=0.0
    )
    with pytest.raises(ValidationError, match="offsets"):
        Transcript(language="en", duration_s=1.0, text="Hello", segments=(seg,))


def test_transcript_rejects_unsupported_language():
    with pytest.raises(ValidationError):
        Transcript(language="fr", duration_s=1.0, text="", segments=())  # pyright: ignore[reportArgumentType]


def test_empty_span_is_rejected():
    with pytest.raises(ValueError, match="span"):
        _transcript().span(1, 1)


def test_report_round_trips_through_json():
    report = SpeechReport(
        source_name="talk.wav",
        model="whisper_small",
        transcript=_transcript(),
        asr_quality=AsrQuality(word_count=2, mean_word_probability=0.9, mean_no_speech_prob=0.01),
    )
    again = SpeechReport.model_validate_json(report.model_dump_json())
    assert again == report
    assert again.schema_version == 6


def test_report_source_is_a_file_name_not_a_path():
    with pytest.raises(ValidationError, match="file name"):
        SpeechReport(
            source_name="C:/Users/someone/talk.wav",
            model="whisper_small",
            transcript=_transcript(),
            asr_quality=AsrQuality(
                word_count=0, mean_word_probability=None, mean_no_speech_prob=None
            ),
        )
