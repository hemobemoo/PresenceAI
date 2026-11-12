"""Facial expression events from blendshapes: labels only, never emotions.

Only the five labels in `ALLOWED_LABELS` may be emitted. Expression events are
informational: they never feed scores, the skill profile or LLM prompts.
Detection is implemented in slice P6.
"""

from __future__ import annotations

from presenceai.schemas.features import ExpressionLabel

ALLOWED_LABELS: frozenset[str] = frozenset(label.value for label in ExpressionLabel)
