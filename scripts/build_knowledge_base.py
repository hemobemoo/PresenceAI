"""Build the knowledge base from the open textbooks (P2-d, ADR 0020; FINAL_07 §F).

    uv run python scripts/build_knowledge_base.py --input data/texts/raw --out data/texts/kb

Checks each EPUB's SHA-256 and licence statement against configs/knowledge.yaml (stops on any
difference), reads it with the standard library, chunks it (~400 e5 tokens), embeds it with
multilingual-e5-small (download first: scripts/download_models.py --only e5_small) and writes
chunks.jsonl, embeddings.npy and meta.json. Prints counts only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from presenceai.evalconfig import REPO_ROOT
from presenceai.knowledge.build import BookCheckError, build, load_settings
from presenceai.knowledge.embed import E5Embedder
from presenceai.speech.asr import ModelNotAvailable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "texts" / "raw")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "texts" / "kb")
    args = ap.parse_args(argv)
    try:
        counts = build(args.input, args.out, E5Embedder(), load_settings())
    except (BookCheckError, ModelNotAvailable) as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1
    for book, n in counts.items():
        print(f"{book}: {n} chunks")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
