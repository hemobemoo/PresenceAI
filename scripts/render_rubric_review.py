"""Write the review document of a rubric draft (ADR 0017).

uv run python scripts/render_rubric_review.py                  # s1_presentation
uv run python scripts/render_rubric_review.py --rubric s1_presentation
"""

from __future__ import annotations

import argparse

from presenceai.evalconfig import REPO_ROOT
from presenceai.rubric import loader, review


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rubric", default="s1_presentation")
    args = ap.parse_args(argv)
    rubric = loader.load_rubric(args.rubric)
    doc = review.review_markdown(rubric, loader.load_thresholds(args.rubric))
    name = args.rubric.split("_")[0].upper()
    out = REPO_ROOT / "docs" / "rubric" / f"{name}_ANCHORS_FOR_REVIEW.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
