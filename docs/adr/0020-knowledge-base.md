# 0020 — Knowledge base from open textbooks, with e5-small embeddings (P2-d)

- Status: **accepted** by the owner on 2026-11-22 (P2-d go-ahead, notes 1–3; source decisions
  the same day).
- Date: 2026-11-22
- Decider: project owner
- Plan reference: FINAL_07 §F (books, rules, build command), FINAL_03 §4 and FINAL_06 §1
  (embeddings), FINAL_00 decision 7 (open texts only)
- Config keys affected: `configs/models.yaml: embeddings` (default now `e5_small`, pinned);
  `configs/knowledge.yaml` (new: chunk size, top-k, spine filter, books with SHA-256 and
  licence)

## Decision

1. **Embeddings: `intfloat/multilingual-e5-small` (MIT) is the default** instead of
   EmbeddingGemma, whose Gemma Terms of Use are not an open-source licence (owner, note 1).
   - Fetched only by `scripts/download_models.py --only e5_small` from huggingface.co at commit
     `614241f622f53c4eeff9890bdc4f31cfecc418b3`. The two files used (`onnx/model.onnx`,
     `onnx/tokenizer.json`) must match both the hash Hugging Face lists for that commit and the
     SHA-256 pinned in `configs/models.yaml`.
   - Run with onnxruntime and `tokenizers`, both already installed (no torch, no new runtime
     dependency). e5's "query: " / "passage: " prefixes; mean pooling; L2-normalised vectors.
   - **RAM on the owner's laptop** (slow test, process RSS): **+737 MB** after loading,
     **+749 MB** after embedding 64 passages (fp32 ONNX). The quantised ONNX export (118 MB)
     needs AVX-512 VNNI and is not used.
2. **Books** (owner, note 2 and the source decisions of 2026-11-22). Each book's EPUB was
   downloaded into the git-ignored `data/texts/raw/`; its SHA-256 and its own licence statement
   (`dc:rights`) are recorded in `configs/knowledge.yaml` and checked before every build, and
   the build **stops** on any difference.

   | Book | Licence (book's own statement) | Source | T2 text |
   |---|---|---|---|
   | *Fundamentals of Public Speaking* (Lumen; FSCJ) | CC BY 4.0 | `openfl.pressbooks.pub/publicspeaking` (moved from `fscj.pressbooks.pub`, which no longer resolves; same book; owner's approval) | yes |
   | *Stand up, Speak out* (UMN) | CC BY-NC-SA 4.0 | `open.lib.umn.edu/publicspeaking` | no |
   | *Speak Out, Call In*, 2nd ed. (Mapes) | CC BY-NC-SA 4.0 | `opentext.ku.edu/speakoutcallin2` (the Open Textbook Library link redirects there) | no |
   | *Exploring Public Speaking*, 4th ed. (Maricopa) | CC BY-NC-SA 4.0 | **not ingested**: no EPUB export (owner: skip for now) | — |

3. **Reading** with the standard library only (`zipfile`, `xml.etree` for the package files,
   `html.parser` for chapters; no ebooklib, which is AGPL). Only spine files whose id starts
   with `chapter` are read (Pressbooks), so covers, front and back matter are skipped. The
   chapter is the table-of-contents label; the section is the latest h2–h4 heading. Scripts,
   navigation and footnote marks are not read.
4. **Chunks** of at most 400 e5 tokens (FINAL_07 §F), never across a section; a longer
   paragraph is split at sentence ends, then at words. **Every chunk keeps book, chapter,
   section and licence**; `t2_allowed` is false for NC-SA books and `t2_training_chunks` never
   returns them (FINAL_07 §F).
5. **Storage and search:** `chunks.jsonl`, `embeddings.npy` and `meta.json` in
   `data/texts/kb/` (git-ignored), instead of the parquet file FINAL_07 sketched (no new
   dependency); cosine search with NumPy; every hit carries a citation with book, chapter,
   section, publisher, licence and URL.

## Result of the first build (owner's laptop, 2026-11-22)

1,804 chunks in 3 min 32 s: *Fundamentals* 352 (the only chunks allowed as T2 text), *Stand
up, Speak out* 1,070, *Speak Out, Call In* 382. Two sample questions returned the matching
chapters first ("How do I reduce filler words?" → "Verbal Delivery"; "How should I structure
the conclusion of a speech?" → "11.1 Why Conclusions Matter"). Retrieval quality is not yet
measured.

## Consequences

- Only one of the three ingested books (Fundamentals, CC BY) can feed T2 training text; the
  NC-SA books are for the knowledge base only.
- The knowledge base is rebuilt locally with `scripts/build_knowledge_base.py`; its content is
  never committed (`data/` is git-ignored).
- What would show this is wrong: retrieval that misses relevant chapters for coach questions
  (to be checked when the coach uses it, P7), or a licence statement changing on a book's page.
