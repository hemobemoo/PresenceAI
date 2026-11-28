"""T1 training, run by t1_run.py in its own process on Colab (ADR 0029 §2).

    python t1train.py train <args.json>   # Unsloth LoRA on DisfluencySpeech train

Its own process, so the GPU is free afterwards and Unsloth's patches never reach the merge
(t1merge.py, plain numpy) or the E3 decoding. Writes only the paths named in args.json; prints no
token (it uses none).
"""

import io
import json
import sys
from pathlib import Path


def decode_16k(data):
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def epoch_of(ckpt):
    state = json.loads((ckpt / "trainer_state.json").read_text(encoding="utf-8"))
    return round(float(state["epoch"]))


def train(a):
    import pyarrow.parquet as pq
    import torch
    import unsloth
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
    )
    from unsloth import FastModel

    hp = a["hp"]
    model, proc = FastModel.from_pretrained(
        model_name=a["src"],
        dtype=None,
        load_in_4bit=False,
        auto_model=WhisperForConditionalGeneration,
        whisper_language="English",
        whisper_task="transcribe",
    )
    lora = hp["lora"]
    model = FastModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=list(lora["target_modules"]),
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=hp["seed"],
        use_rslora=False,
        loftq_config=None,
        task_type=None,  # required for Whisper (Unsloth notebook)
    )
    model.generation_config.language = "<|en|>"
    model.generation_config.task = "transcribe"
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    rows, dropped = [], 0
    for r in pq.read_table(a["data"]).to_pylist():
        ids = proc.tokenizer(r["text"]).input_ids
        if len(ids) > hp["max_label_tokens"]:
            dropped += 1
            continue
        rows.append((r["audio"], ids))

    class Clips(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            audio, ids = rows[i]
            feats = proc.feature_extractor(decode_16k(audio), sampling_rate=16000)
            return {"input_features": feats.input_features[0], "labels": ids}

    def collate(features):
        batch = proc.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        lab = proc.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == proc.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

    opt, bt = hp["optimiser"], hp["batch"]
    args = Seq2SeqTrainingArguments(
        output_dir=a["ckpt_dir"],
        per_device_train_batch_size=bt["per_device"],
        gradient_accumulation_steps=bt["gradient_accumulation"],
        warmup_steps=opt["warmup_steps"],
        num_train_epochs=hp["epochs"],
        learning_rate=opt["learning_rate"],
        lr_scheduler_type=opt["lr_scheduler"],
        weight_decay=opt["weight_decay"],
        optim=opt["optim"],
        fp16=hp["precision"] == "fp16",
        bf16=hp["precision"] == "bf16",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=None,  # every epoch is a pick candidate
        eval_strategy="no",  # the pick decodes with the app's settings afterwards (ADR 0029 §5)
        seed=hp["seed"],
        data_seed=hp["seed"],
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=Clips(),
        data_collator=collate,
        processing_class=proc.feature_extractor,
        args=args,
    )
    resume = any(Path(a["ckpt_dir"]).glob("checkpoint-*"))
    out = trainer.train(resume_from_checkpoint=True if resume else None)

    import peft
    import transformers

    ckpts = sorted(
        Path(a["ckpt_dir"]).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
    )
    report = {
        "rows_used": len(rows),
        "rows_dropped_label_too_long": dropped,
        "resumed": resume,
        "global_step": out.global_step,
        "train_loss": out.training_loss,
        "log_history": trainer.state.log_history,
        "checkpoints": [{"name": c.name, "epoch": epoch_of(c)} for c in ckpts],
        "versions": {
            "unsloth": unsloth.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
        },
    }
    Path(a["report"]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    {"train": train}[mode](json.loads(Path(path).read_text(encoding="utf-8")))
