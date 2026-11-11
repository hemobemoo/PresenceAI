---
description: Run all checks and report only failures
---

Run, in order, and stop at the first group that fails:

1. `uv run ruff check .` and `uv run ruff format --check .`
2. `uv run pyright`
3. `uv run pytest -m "not slow" -q`
4. `uv run pytest tests/safety -q`
5. `uv run python scripts/check_licenses.py`

Report only failures with file and line, and a one-line proposed fix for each. Do not change code unless I ask.
