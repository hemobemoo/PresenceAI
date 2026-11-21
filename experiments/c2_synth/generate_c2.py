"""Generate C2 transcripts on a free Colab T4 (P2-e, ADR 0021). Colab only, never in the app.

    python generate_c2.py fetch  --model-dir /content/qwen35-9b
    python generate_c2.py run    --model-dir /content/qwen35-9b --prompts prompts.jsonl \
                                 --out /content/drive/MyDrive/presenceai_c2 [--ids a,b] [--batch 5]

- **Model:** Qwen/Qwen3.5-9B (Apache-2.0: model card and LICENSE file) at the pinned commit
  below, loaded in 4-bit NF4 with bitsandbytes (fp16 compute; the T4 has no bf16).
- **fetch** downloads the files and checks every one against the hash Hugging Face lists for that
  commit (SHA-256 for large files, the git blob SHA-1 for small ones); any mismatch stops.
  **run** checks them again before loading.
- **Quality loop** (owner, 2026-11-22, after pilot 2b). Each attempt is accepted only if:
  1. it ended before the 800-token cap, a safety net only; a capped transcript is always
     rejected (a cut-off talk would be an unplanned "no conclusion");
  2. it has 300-550 words and no markdown, heading, bullet, label or thinking text;
  3. it does not echo the prompt ("main idea", "central idea").
  The model's own yes/no answers are **not** used: in pilot 2b it answered "attention-catching
  opening = yes" for 4 plain "Hello everyone" openings, so it could not be trusted as a filter.
  Nothing here uses PresenceAI's cue lexicons or text detectors. A failed item is regenerated
  with a new seed, at most 3 attempts; items that still fail are logged and left out. The
  planted labels stay unverified except for the owner's hand-checked sample (ADR 0021).
- **Files in --out** (resumable; each line is flushed to disk): `generated.jsonl` (accepted),
  `rejected.jsonl` (failed 3 times), `attempts.jsonl` (every attempt with its result, for the
  rejection count per group). A new run skips ids already accepted or rejected.
- Only synthetic prompts (resources/c2/prompts.jsonl) and the generated transcripts are used.
- Non-thinking mode (`enable_thinking=False`); sampling as the model card recommends for
  instruct use (temperature 0.7, top-p 0.8, top-k 20), seeded per prompt id and attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PATTERNS = ["*.safetensors", "*.json", "*.jinja", "merges.txt", "vocab.json", "LICENSE"]
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
MAX_NEW_TOKENS = 800
MAX_ATTEMPTS = 3
MIN_WORDS, MAX_WORDS = 300, 550
ECHO = re.compile(r"\b(main|central) idea\b", re.IGNORECASE)
FORMAT = re.compile(
    r"(^\s*#)|(^\s*[-*•]\s)|(^\s*\d+[.)]\s)|(\*\*)|(<think>)|(</think>)|(^\s*\[[^\]]+\])"
    r"|(^\s*\([^)]*\)\s*$)|(^\s*(speaker|presenter|student)\s*\d*\s*:)",
    re.IGNORECASE | re.MULTILINE,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expected_hashes() -> dict[str, tuple[str, str]]:
    url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as resp:  # fixed https URL
        listing = json.loads(resp.read().decode("utf-8"))
    out: dict[str, tuple[str, str]] = {}
    for f in listing:
        if f.get("type") != "file":
            continue
        lfs = f.get("lfs")
        out[f["path"]] = ("sha256", lfs["oid"]) if lfs else ("git-sha1", f["oid"])
    return out


def _digest(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        h = hashlib.sha256()
    else:  # git blob SHA-1, as Hugging Face lists for small files
        h = hashlib.sha1(usedforsecurity=False)
        h.update(f"blob {path.stat().st_size}".encode() + b"\x00")
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(model_dir: Path) -> int:
    expected = expected_hashes()
    files = sorted(p for p in model_dir.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if not any(p.suffix == ".safetensors" for p in files):
        raise SystemExit(f"no weights in {model_dir}; run `fetch` first")
    for p in files:
        rel = p.relative_to(model_dir).as_posix()
        if rel not in expected:
            raise SystemExit(f"{rel}: not in {REPO}@{REVISION[:7]}")
        algorithm, want = expected[rel]
        if _digest(p, algorithm) != want:
            raise SystemExit(f"{rel}: {algorithm} does not match {REPO}@{REVISION[:7]}")
    _log(f"verified {len(files)} files against {REPO}@{REVISION[:7]}")
    return len(files)


def fetch(model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, revision=REVISION, local_dir=str(model_dir), allow_patterns=PATTERNS)
    verify(model_dir)


def text_problems(text: str) -> list[str]:
    """Length, format and prompt-echo checks (no cue lexicons, no detectors)."""
    out = []
    n = len(text.split())
    if not MIN_WORDS <= n <= MAX_WORDS:
        out.append(f"words {n}")
    if FORMAT.search(text):
        out.append("format")
    if ECHO.search(text):
        out.append("echo")
    return out


def _read_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(x)["id"] for x in path.read_text("utf-8").splitlines() if x.strip()}


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load(model_dir: Path) -> tuple[Any, Any]:
    import torch
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )  # fmt: skip
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    model = None
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, name).from_pretrained(
                model_dir, quantization_config=quant, device_map="cuda:0", dtype=torch.float16
            )
            _log(f"loaded with {name}")
            break
        except (ValueError, KeyError) as exc:
            _log(f"{name} cannot load it ({type(exc).__name__}); trying the next class")
    if model is None:
        raise SystemExit("no transformers class could load the model")
    model.eval()
    return tok, model


class Generator:
    def __init__(self, tok: Any, model: Any) -> None:
        self.tok, self.model = tok, model

    def _chat(self, contents: list[str]) -> Any:
        texts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for c in contents
        ]  # fmt: skip
        return self.tok(texts, return_tensors="pt", padding=True).to("cuda:0")

    def write(self, prompts: list[str], seed: int) -> list[tuple[str, int, bool]]:
        """(text, new tokens, reached the cap) per prompt."""
        import torch

        enc = self._chat(prompts)
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, **SAMPLING)
        out = []
        cfg_eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos = {self.tok.eos_token_id}
        eos |= set(cfg_eos) if isinstance(cfg_eos, list) else {cfg_eos}
        for ids in gen[:, enc["input_ids"].shape[1] :]:
            real = [int(i) for i in ids if int(i) != self.tok.pad_token_id]
            ended = any(i in eos for i in ids.tolist())
            text = self.tok.decode(ids, skip_special_tokens=True).strip()
            out.append((text, len(real), not ended and len(real) >= MAX_NEW_TOKENS))
        return out


def run(args: argparse.Namespace) -> None:
    prompts = [json.loads(x) for x in Path(args.prompts).read_text("utf-8").splitlines() if x]
    if args.ids:
        wanted = set(args.ids.split(","))
        prompts = [p for p in prompts if p["id"] in wanted]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    finished = _read_ids(out / "generated.jsonl") | _read_ids(out / "rejected.jsonl")
    todo = [p for p in prompts if p["id"] not in finished]
    _log(f"{len(prompts)} prompts, {len(prompts) - len(todo)} finished, {len(todo)} to go")
    if not todo:
        return
    verify(Path(args.model_dir))
    gen = Generator(*_load(Path(args.model_dir)))
    meta = {"model": REPO, "revision": REVISION, "quantization": "bnb-nf4-4bit-fp16",
            "sampling": SAMPLING, "max_new_tokens": MAX_NEW_TOKENS}  # fmt: skip
    attempts = dict.fromkeys((p["id"] for p in todo), 0)
    if (out / "attempts.jsonl").is_file():  # resume: earlier sessions' attempts still count
        for line in (out / "attempts.jsonl").read_text("utf-8").splitlines():
            row = json.loads(line) if line.strip() else {}
            if row.get("id") in attempts:
                attempts[row["id"]] = max(attempts[row["id"]], int(row["attempt"]))
    queue = list(todo)
    while queue:
        batch, queue = queue[: args.batch], queue[args.batch :]
        attempt_no = {p["id"]: attempts[p["id"]] + 1 for p in batch}
        key = f"{batch[0]['id']}:{attempt_no[batch[0]['id']]}"
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        t0 = time.time()
        written = gen.write([p["prompt"] for p in batch], seed)
        seconds = round(time.time() - t0, 1)
        for p, (text, n, capped) in zip(batch, written, strict=True):
            attempts[p["id"]] = attempt_no[p["id"]]
            problems = ["cap"] if capped else []
            problems += text_problems(text)
            row = {"id": p["id"], "group": p["group"], "hook": p["hook"], "part": p["part"],
                   "attempt": attempts[p["id"]], "new_tokens": n, "words": len(text.split()),
                   "problems": problems, "batch_seconds": seconds, "seed": seed}  # fmt: skip
            _append(out / "attempts.jsonl", {**row, "text": text})  # synthetic; for auditing
            if not problems:
                _append(out / "generated.jsonl", {**row, "text": text, **meta})
            elif attempts[p["id"]] >= MAX_ATTEMPTS:
                _append(out / "rejected.jsonl", {**row, "text": text, **meta})
            else:
                queue.append(p)
        _log(f"batch of {len(batch)} in {seconds} s; {len(queue)} waiting")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["fetch", "verify", "run"])
    ap.add_argument("--model-dir", default="/content/qwen35-9b")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--out", default="/content/drive/MyDrive/presenceai_c2")
    ap.add_argument("--ids", default="", help="comma-separated prompt ids (the pilot)")
    ap.add_argument("--batch", type=int, default=5)
    args = ap.parse_args(argv)
    if args.action == "fetch":
        fetch(Path(args.model_dir))
    elif args.action == "verify":
        verify(Path(args.model_dir))
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
