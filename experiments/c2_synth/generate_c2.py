"""Generate C2 transcripts on a free Colab T4 (P2-e, ADR 0021). Runs on Colab only, never in the app.

    python generate_c2.py fetch  --model-dir /content/qwen35-9b
    python generate_c2.py run    --model-dir /content/qwen35-9b --prompts prompts.jsonl \
                                 --out /content/drive/MyDrive/presenceai_c2 [--ids a,b] [--batch 4]

- **Model:** Qwen/Qwen3.5-9B (Apache-2.0: model card and LICENSE file) at the pinned commit
  below, loaded in 4-bit NF4 with bitsandbytes (fp16 compute; the T4 has no bf16).
- **fetch** downloads the files and checks every one against the hash Hugging Face lists for that
  commit (SHA-256 for large files, the git blob SHA-1 for small ones); any mismatch stops.
  **run** checks them again before loading.
- **Resumable:** every finished batch is appended to `<out>/generated.jsonl` and flushed to disk;
  a new run skips ids already there. Put `--out` on Google Drive so a stopped session loses
  nothing.
- Only synthetic prompts (resources/c2/prompts.jsonl) and the generated transcripts are used.
- Non-thinking mode (`enable_thinking=False`); sampling as the model card recommends for
  instruct use (temperature 0.7, top-p 0.8, top-k 20), seeded per prompt id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PATTERNS = ["*.safetensors", "*.json", "*.jinja", "merges.txt", "vocab.json", "LICENSE"]
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expected_hashes() -> dict[str, tuple[str, str]]:
    url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
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


def _done(out: Path) -> set[str]:
    path = out / "generated.jsonl"
    if not path.is_file():
        return set()
    return {json.loads(line)["id"] for line in path.read_text("utf-8").splitlines() if line.strip()}


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


def run(args: argparse.Namespace) -> None:
    import torch

    prompts = [json.loads(x) for x in Path(args.prompts).read_text("utf-8").splitlines() if x]
    if args.ids:
        wanted = set(args.ids.split(","))
        prompts = [p for p in prompts if p["id"] in wanted]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    todo = [p for p in prompts if p["id"] not in done]
    _log(
        f"{len(prompts)} prompts, {len(done & {p['id'] for p in prompts})} done, {len(todo)} to go"
    )
    if not todo:
        return
    verify(Path(args.model_dir))
    tok, model = _load(Path(args.model_dir))
    meta = {"model": REPO, "revision": REVISION, "quantization": "bnb-nf4-4bit-fp16",
            "sampling": SAMPLING, "max_new_tokens": args.max_new_tokens}  # fmt: skip
    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            for p in batch
        ]  # fmt: skip
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        seed = int(hashlib.sha256(batch[0]["id"].encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, **SAMPLING)
        seconds = time.time() - t0
        new = gen[:, enc["input_ids"].shape[1] :]
        with (out / "generated.jsonl").open("a", encoding="utf-8") as fh:
            for p, ids in zip(batch, new, strict=True):
                text = tok.decode(ids, skip_special_tokens=True).strip()
                n = int((ids != tok.pad_token_id).sum())
                fh.write(json.dumps({"id": p["id"], "text": text, "new_tokens": n,
                                     "batch_seconds": round(seconds, 1), "seed": seed, **meta},
                                    ensure_ascii=False) + "\n")  # fmt: skip
            fh.flush()
            os.fsync(fh.fileno())
        _log(f"batch {i // args.batch + 1}: {len(batch)} done in {seconds:.0f} s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["fetch", "verify", "run"])
    ap.add_argument("--model-dir", default="/content/qwen35-9b")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--out", default="/content/drive/MyDrive/presenceai_c2")
    ap.add_argument("--ids", default="", help="comma-separated prompt ids (the pilot)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=700)
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
