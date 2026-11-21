"""Text embeddings for the knowledge base: multilingual-e5-small with onnxruntime (ADR 0020).

The model is fetched only by `scripts/download_models.py --only e5_small` (pinned SHA-256) and
run with onnxruntime and the `tokenizers` package (no torch). e5 expects the prefixes "query: "
and "passage: "; vectors are mean-pooled over the attention mask and L2-normalised, so the dot
product is the cosine similarity. Tests use a fake embedder with the same interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from presenceai.speech.asr import MODELS_ROOT, ModelNotAvailable

E5_DIR = MODELS_ROOT / "multilingual-e5-small"
MAX_TOKENS = 512
Kind = Literal["query", "passage"]


class Embedder(Protocol):
    name: str

    def count_tokens(self, text: str) -> int: ...

    def embed(self, texts: Sequence[str], kind: Kind) -> np.ndarray: ...


class E5Embedder:
    name = "intfloat/multilingual-e5-small"

    def __init__(self, folder: Path = E5_DIR, batch: int = 16) -> None:
        model, tok = folder / "onnx" / "model.onnx", folder / "onnx" / "tokenizer.json"
        if not model.is_file() or not tok.is_file():
            raise ModelNotAvailable(
                f"{folder.name} is not downloaded; run: "
                "uv run python scripts/download_models.py --only e5_small"
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tok))
        self._tok.enable_truncation(MAX_TOKENS)
        self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self._batch = batch

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(f"passage: {text}", add_special_tokens=True).ids)

    def embed(self, texts: Sequence[str], kind: Kind) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch):
            enc = self._tok.encode_batch([f"{kind}: {t}" for t in texts[i : i + self._batch]])
            width = max(len(e.ids) for e in enc)
            ids = np.zeros((len(enc), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, e in enumerate(enc):
                ids[row, : len(e.ids)] = e.ids
                mask[row, : len(e.ids)] = 1
            (raw,) = self._session.run(
                ["last_hidden_state"],
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
            )
            hidden = np.asarray(raw, dtype=np.float32)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)
