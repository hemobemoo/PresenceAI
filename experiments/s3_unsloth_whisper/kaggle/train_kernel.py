"""Spike S3 Kaggle script kernel: LoRA fine-tune of Whisper-small with Unsloth (ADR 0005).

Adapted from unslothai/notebooks nb/Kaggle-Whisper.ipynb at the commit in CONFIG
(source_notebook). Changes from the notebook, all recorded in run_info.json:
- base model whisper-small (notebook: whisper-large-v3), pinned by revision;
- data: DisfluencySpeech (pinned revision), `transcript_a`, 200 train / 50 validation clips
  chosen with a fixed seed (notebook: Etherll/kaira);
- validation loss only (no argmax WER metric), evaluated at step 0 and every eval_steps;
- max_steps 100 and eval_steps 10;
- after training: LoRA merged to 16-bit, base and fine-tuned models converted to CTranslate2,
  and 10 test clips written as WAV for the local CPU check.

push.py replaces __CONFIG_JSON__ before upload. Nothing here reads a secret: the model and
the dataset are public. Runs only on Kaggle (unsloth, torch and ctranslate2 are installed below).
"""

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

CONFIG = json.loads(r"""__CONFIG_JSON__""")
OUT = "/kaggle/working/s3"
INFO: dict = {"config": CONFIG, "started_unix": time.time(), "changes_from_notebook": __doc__}


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)  # noqa: S602 (fixed commands, Kaggle only)


def versions() -> dict:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in (
        "torch",
        "transformers",
        "unsloth",
        "unsloth_zoo",
        "peft",
        "trl",
        "datasets",
        "ctranslate2",
        "accelerate",
        "bitsandbytes",
        "librosa",
        "soundfile",
    ):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def install() -> None:
    # Notebook cell 1 (same packages and pins), plus ctranslate2 for the conversion.
    sh("pip install -q pip3-autoremove")
    sh(
        "pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128"
    )
    sh("pip install -q unsloth")
    sh('pip install -q --no-deps --upgrade "torchao>=0.16.0"')
    sh("pip install -q transformers==4.57.6")
    sh("pip install -q --no-deps trl==0.22.2")
    import importlib.metadata as md

    try:
        tv = tuple(int(x) for x in md.version("torch").split("+")[0].split(".")[:2])
    except Exception:
        tv = (2, 11)
    codec = {
        (2, 11): "torchcodec>=0.11.0",
        (2, 10): "torchcodec==0.10.*",
        (2, 9): "torchcodec==0.9.*",
        (2, 8): "torchcodec==0.7.*",
        (2, 7): "torchcodec==0.5.*",
    }.get(tv, "torchcodec>=0.12.0" if tv >= (2, 12) else "torchcodec")
    sh(f'pip install -q librosa soundfile evaluate jiwer "{codec}" "datasets>=3.4.1,<4.0.0"')
    sh("pip install -q ctranslate2")


def pick(split: str, n: int):
    from datasets import Audio, load_dataset

    ds = load_dataset(CONFIG["dataset"]["id"], revision=CONFIG["dataset"]["revision"], split=split)
    ds = ds.shuffle(seed=CONFIG["seeds"]["data"]).select(range(n))
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    install()
    INFO["versions_after_install"] = versions()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration
    from unsloth import FastModel, is_bf16_supported

    INFO["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    base_dir = snapshot_download(
        CONFIG["base_model"]["id"], revision=CONFIG["base_model"]["revision"]
    )
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_dir,
        dtype=None,
        load_in_4bit=False,
        auto_model=WhisperForConditionalGeneration,
        whisper_language="English",
        whisper_task="transcribe",
    )
    lc = CONFIG["lora"]
    model = FastModel.get_peft_model(
        model,
        r=lc["r"],
        target_modules=lc["target_modules"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seeds"]["lora"],
        use_rslora=False,
        loftq_config=None,
        task_type=None,
    )
    model.generation_config.language = "<|en|>"
    model.generation_config.task = "transcribe"
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    col = CONFIG["dataset"]["text_column"]

    def fmt(ex: dict) -> dict:
        feats = tokenizer.feature_extractor(
            ex["audio"]["array"], sampling_rate=ex["audio"]["sampling_rate"]
        )
        return {
            "input_features": feats.input_features[0],
            "labels": tokenizer.tokenizer(ex[col]).input_ids,
        }

    d = CONFIG["dataset"]
    train_raw, val_raw, test_raw = (
        pick("train", d["n_train"]),
        pick("validation", d["n_validation"]),
        pick("test", d["n_test"]),
    )
    train_ds, val_ds = [fmt(x) for x in train_raw], [fmt(x) for x in val_raw]

    class Collator:  # notebook's DataCollatorSpeechSeq2SeqWithPadding
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features):
            batch = self.processor.feature_extractor.pad(
                [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
            )
            lab = self.processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
            )
            labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    t = CONFIG["train"]
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tokenizer),
        processing_class=tokenizer.feature_extractor,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            warmup_steps=t["warmup_steps"],
            max_steps=t["max_steps"],
            learning_rate=t["learning_rate"],
            logging_steps=t["logging_steps"],
            optim=t["optim"],
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            weight_decay=t["weight_decay"],
            remove_unused_columns=False,
            lr_scheduler_type=t["lr_scheduler_type"],
            label_names=["labels"],
            eval_steps=t["eval_steps"],
            eval_strategy="steps",
            seed=CONFIG["seeds"]["trainer"],
            output_dir="/kaggle/working/outputs",
            save_strategy="no",
            report_to="none",
        ),
    )
    eval0 = trainer.evaluate()
    stats = trainer.train()
    INFO["eval_loss_step0"] = eval0.get("eval_loss")
    INFO["train_runtime_s"] = stats.metrics.get("train_runtime")
    INFO["peak_gpu_mem_gb"] = (
        round(torch.cuda.max_memory_reserved() / 1024**3, 3) if torch.cuda.is_available() else None
    )

    rows = [
        {
            "step": 0,
            "loss": "",
            "eval_loss": eval0.get("eval_loss"),
            "learning_rate": "",
            "epoch": 0,
        }
    ]
    for h in trainer.state.log_history:
        if "loss" in h or "eval_loss" in h:
            rows.append(
                {
                    "step": h.get("step"),
                    "loss": h.get("loss", ""),
                    "eval_loss": h.get("eval_loss", ""),
                    "learning_rate": h.get("learning_rate", ""),
                    "epoch": h.get("epoch", ""),
                }
            )
    with open(f"{OUT}/train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    merged = "/kaggle/working/merged"
    try:
        model.save_pretrained_merged(merged, tokenizer, save_method=None)
        INFO["merge_method"] = "unsloth save_pretrained_merged(save_method=None)"
    except Exception as exc:  # fall back to plain PEFT merge
        INFO["merge_unsloth_error"] = repr(exc)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged)
        tokenizer.save_pretrained(merged)
        INFO["merge_method"] = "peft merge_and_unload"
    q = CONFIG["kaggle"]["ct2_quantization"]
    copy = "--copy_files tokenizer.json preprocessor_config.json"
    conv = "ct2-transformers-converter"
    sh(f"{conv} --model {merged} --output_dir {OUT}/ct2_ft --quantization {q} {copy}")
    sh(f"{conv} --model {base_dir} --output_dir {OUT}/ct2_base --quantization {q} {copy}")

    import soundfile as sf

    os.makedirs(f"{OUT}/clips", exist_ok=True)
    refs = []
    for i, ex in enumerate(test_raw):
        name = f"test_{i:02d}.wav"
        sf.write(f"{OUT}/clips/{name}", ex["audio"]["array"], 16000)
        refs.append(
            {
                "clip": name,
                "duration_s": round(len(ex["audio"]["array"]) / 16000, 2),
                "reference": ex[col],
            }
        )
    with open(f"{OUT}/clips/refs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "duration_s", "reference"])
        w.writeheader()
        w.writerows(refs)
    shutil.rmtree(merged, ignore_errors=True)
    shutil.rmtree("/kaggle/working/outputs", ignore_errors=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
        INFO["status"] = "ok"
    except BaseException:
        INFO["status"] = "error"
        INFO["traceback"] = traceback.format_exc()
        code = 1
    INFO["finished_unix"] = time.time()
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/run_info.json", "w", encoding="utf-8") as f:
        json.dump(INFO, f, indent=2, default=str)
    print(
        json.dumps(
            {k: INFO.get(k) for k in ("status", "gpu", "eval_loss_step0", "train_runtime_s")}
        ),
        flush=True,
    )
    sys.exit(code)  # the script ends here, so the Kaggle session (and its GPU) stops
