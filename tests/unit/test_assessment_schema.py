from typing import Any

import pytest
from pydantic import ValidationError

from presenceai.schemas.assessment import (
    Confidence,
    Evidence,
    Measurement,
    TimeSpan,
    TranscriptSpan,
)


def _measurement(**overrides: Any) -> Measurement:
    data: dict[str, Any] = {
        "value": 148.0,
        "unit": "wpm",
        "confidence": "high",
        "method": "words / speaking time",
        "evidence": [{"time": {"start_s": 72.0, "end_s": 100.0}}],
        "language": "en",
        "live": False,
    }
    data.update(overrides)
    return Measurement.model_validate(data)


def test_valid_measurement():
    m = _measurement()
    assert m.confidence is Confidence.HIGH
    assert m.evidence[0].time == TimeSpan(start_s=72.0, end_s=100.0)


def test_evidence_required():
    with pytest.raises(ValidationError):
        _measurement(evidence=[])


def test_evidence_needs_a_span():
    with pytest.raises(ValidationError):
        Evidence()


def test_evidence_can_have_both():
    e = Evidence(
        transcript=TranscriptSpan(start_char=0, end_char=5, text="hello"),
        time=TimeSpan(start_s=0, end_s=0.4),
    )
    assert e.transcript is not None
    assert e.time is not None


@pytest.mark.parametrize(("start", "end"), [(5.0, 4.0), (-1.0, 2.0)])
def test_time_span_ordering(start: float, end: float):
    with pytest.raises(ValidationError):
        TimeSpan(start_s=start, end_s=end)


def test_transcript_span_ordering():
    with pytest.raises(ValidationError):
        TranscriptSpan(start_char=4, end_char=4, text="x")


@pytest.mark.parametrize(("field", "value"), [("confidence", "certain"), ("language", "fr")])
def test_rejects_unknown_enum_values(field: str, value: str):
    with pytest.raises(ValidationError):
        _measurement(**{field: value})


def test_rejects_extra_fields():
    with pytest.raises(ValidationError):
        _measurement(emotion="happy")


def test_frozen():
    m = _measurement()
    with pytest.raises(ValidationError):
        m.value = 1.0  # pyright: ignore[reportAttributeAccessIssue]
