"""Merge a LoRA adapter into the base weights (ADR 0029 §3). Pure numpy, so it is tested here.

W' = W + (B @ A) * alpha / r for every adapted linear layer; every other tensor is the base's own,
so the merged file has exactly the base's weight names, shapes and dtypes and the app's converter
sees the app's model with changed attention weights only. No PEFT or transformers version is
involved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PREFIXES = ("base_model.model.",)


def _module(key: str) -> tuple[str, str] | None:
    """('model.encoder.layers.0.self_attn.q_proj', 'A' | 'B') for an adapter key, else None."""
    for part in ("lora_A", "lora_B"):
        for tail in (f".{part}.weight", f".{part}.default.weight"):
            if key.endswith(tail):
                mod = key[: -len(tail)]
                for p in PREFIXES:
                    if mod.startswith(p):
                        mod = mod[len(p) :]
                return mod, part[-1]
    return None


def check_config(cfg: Mapping[str, Any], r: int, alpha: int, targets: Sequence[str]) -> float:
    """The adapter must be plain LoRA with the run's rank and alpha; returns the scaling."""
    problems = []
    if cfg.get("r") != r or cfg.get("lora_alpha") != alpha:
        problems.append(f"r/alpha {cfg.get('r')}/{cfg.get('lora_alpha')} != {r}/{alpha}")
    for flag in ("use_rslora", "use_dora"):
        if cfg.get(flag):
            problems.append(f"{flag} is on")
    if cfg.get("bias", "none") != "none":
        problems.append(f"bias {cfg.get('bias')}")
    tm = cfg.get("target_modules")  # a list, or a regex string (then merge() checks each name)
    if isinstance(tm, list) and sorted(tm) != sorted(targets):
        problems.append(f"target modules {tm} != {list(targets)}")
    if problems:
        raise ValueError("adapter is not the run's LoRA: " + "; ".join(problems))
    return alpha / r


def merge(
    base: Mapping[str, np.ndarray],
    adapter: Mapping[str, np.ndarray],
    scaling: float,
    targets: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """(merged tensors with the base's names, the base keys that changed)."""
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for k, v in adapter.items():
        m = _module(k)
        if m is None:
            raise ValueError(f"unexpected adapter tensor {k}")
        pairs.setdefault(m[0], {})[m[1]] = v
    if not pairs:
        raise ValueError("the adapter holds no LoRA weights")
    out = dict(base)
    changed = []
    for mod, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise ValueError(f"{mod}: lora_A and lora_B must both be present")
        if mod.rsplit(".", 1)[-1] not in targets:
            raise ValueError(f"{mod} is not a target module {list(targets)}")
        key = f"{mod}.weight"
        if key not in base:
            raise ValueError(f"{key} is not a base weight")
        w = base[key]
        delta = ab["B"].astype(np.float32) @ ab["A"].astype(np.float32)
        if delta.shape != w.shape:
            raise ValueError(f"{key}: LoRA shape {delta.shape} != base {w.shape}")
        out[key] = (w.astype(np.float32) + scaling * delta).astype(w.dtype)
        changed.append(key)
    return out, changed
