# Spike S2 — faster-whisper on a ZeroGPU Space

Question, setup, pass rule and results: `docs/adr/0007-zerogpu-space-spike.md`.

| File | Role |
|---|---|
| `build_clip.py` | 3-minute and 10-second clips from public DisfluencySpeech test audio (pinned revision, seed) |
| `prepare.py` | `model`: SHA-256 manifest of the converted model and a model card; `space`: builds `data/s2/space` (app, copy of the shared audio loader, requirements, card, config) |
| `space/app.py` | The Space: `gpu` endpoint (probe or full clip inside `@spaces.GPU`), `cpu` endpoint (whisper-small fallback), `env` endpoint (versions) |
| `measure.py` | Calls the private Space with `gradio_client` (experiment-only exception to AGENTS rule 2), records each call, resumes after a quota reset, applies the pass rule |
| `common.py` | Config, paths, checksums, pass rule (unit-tested in `tests/unit/test_s2_zerogpu.py`) |
| `../configs/s2_zerogpu.yaml` | Model and dataset revisions, Space requirements, decoding options, pass rule |

Kept out of git: the converted model (`models/s2/`), the clips and the built Space folder
(`data/s2/`). Repo ids and the user name are passed on the command line or set as Space
variables; `HF_TOKEN` is used by variable name only.

## Commands

```bash
uv run --with pyarrow python experiments/s2_zerogpu/build_clip.py
uv run python experiments/s2_zerogpu/prepare.py model
uv run python experiments/s2_zerogpu/prepare.py space        # after pinning model.repo_revision
uv run python experiments/s2_zerogpu/measure.py probe --space <owner>/<space> --run-id <id>
uv run python experiments/s2_zerogpu/measure.py run   --space <owner>/<space> --run-id <id>
uv run python experiments/s2_zerogpu/measure.py cpu   --space <owner>/<space> --run-id <id>
```
