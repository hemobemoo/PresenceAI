"""Spike S3 Colab script: LoRA fine-tune of Whisper-small with Unsloth (ADR 0005).

colab.py build injects the config and the build info and puts this script in a notebook cell
(%%writefile /content/train_s3.py), which runs it as `python /content/train_s3.py` on a Colab T4
runtime. Outputs go to /content/s3. Nothing here reads a secret: the model and the dataset are
public. Runs only on Colab (unsloth, torch and ctranslate2 are installed below).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU only, set before torch is imported

import csv
import hashlib
import importlib.metadata as md
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback

CONFIG = json.loads(r"""__CONFIG_JSON__""")
BUILD = json.loads(r"""__BUILD_JSON__""")
OUT = "/content/s3"
WORK = "/content/s3_work"  # merged model and trainer output; deleted before the end

CHANGES_FROM_NOTEBOOK = """\
Adapted from unslothai/notebooks nb/Whisper.ipynb (Colab) at the commit in
CONFIG["source_notebook"]. Its code cells are the same as nb/Kaggle-Whisper.ipynb at that commit
except the install cell; this script uses the Colab install cell. Changes from the notebook:
- base model whisper-small (notebook: whisper-large-v3), pinned by revision;
- data: DisfluencySpeech (pinned revision), `transcript_a`, 200 train / 50 validation clips
  chosen with a fixed seed (notebook: Etherll/kaira);
- validation loss only (no argmax WER metric), evaluated at step 0 and every eval_steps;
- max_steps 100 and eval_steps 10;
- CUDA_VISIBLE_DEVICES=0;
- `mslk` is uninstalled only if it is installed and fails to import (Kaggle run 1 failed on
  this; see env_fixes);
- if the merge writes no config.json / generation_config.json, they are copied from the base
  model (run c1 failed at conversion without them; see env_fixes);
- the base snapshot has no tokenizer.json, so ct2_base gets the fine-tuned export's
  tokenizer.json, which is the same tokenizer (run c2 failed here; see env_fixes);
- after training: LoRA merged to 16-bit, base and fine-tuned models converted to CTranslate2,
  10 test clips written as WAV for the local CPU check, SHA-256 of every output file recorded.
"""

INFO: dict = {
    "config": CONFIG,
    "build": BUILD,
    "changes_from_notebook": CHANGES_FROM_NOTEBOOK,
    "started_unix": time.time(),
    "env_fixes": [],
}


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)  # noqa: S602 (fixed commands, Colab only)


def on_colab() -> bool:
    return any(k.startswith("COLAB_") for k in os.environ)  # the notebook's own test


def versions() -> dict:
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
        "torchao",
        "librosa",
        "soundfile",
    ):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def runtime_info() -> dict:
    info = {
        "on_colab": on_colab(),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        q = ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        info["nvidia_smi"] = subprocess.run(  # noqa: S603 (fixed command)
            ["nvidia-smi", *q],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        info["nvidia_smi"] = repr(exc)
    return info


def install() -> None:
    # Notebook install cell (same packages and pins), plus ctranslate2 for the conversion.
    if not on_colab():
        sh("pip install -q unsloth")  # the notebook's non-Colab branch
    else:
        sh(
            'pip install -q sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0"'
            " hf_transfer"
        )
        sh("pip install -q --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth")
        sh("unsloth install-kernels")
        sh('pip install -q --no-deps --upgrade "torchao>=0.16.0"')
    sh("pip install -q transformers==4.57.6")
    sh("pip install -q --no-deps trl==0.22.2")
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
    # Conditional fix (Kaggle run 1, 2026-11-14): an installed `mslk` whose library does not load
    # breaks `import unsloth` (unsloth -> torchao -> mslk). Remove it only in that case.
    if importlib.util.find_spec("mslk") is not None:
        probe = subprocess.run([sys.executable, "-c", "import mslk"], capture_output=True)
        if probe.returncode != 0:
            sh("pip uninstall -y -q mslk")
            INFO["env_fixes"].append("pip uninstall mslk (installed, failed to import)")


def pick(split: str, n: int):
    from datasets import Audio, load_dataset

    ds = load_dataset(CONFIG["dataset"]["id"], revision=CONFIG["dataset"]["revision"], split=split)
    ds = ds.shuffle(seed=CONFIG["seeds"]["data"]).select(range(n))
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checksums(root: str) -> dict:
    """SHA-256 of every file under root except run_info.json (which holds this table)."""
    out = {}
    for d, _, files in os.walk(root):
        for name in files:
            p = os.path.join(d, name)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel != "run_info.json":
                out[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    return dict(sorted(out.items()))


def fill_missing_config(merged: str, base_dir: str) -> list:
    """Copy config.json and generation_config.json from the base model when the merge wrote
    none (run c1: Unsloth save_pretrained_merged(save_method=None) wrote weights and tokenizer
    only, so the CTranslate2 converter could not recognise the model). A LoRA merge does not
    change the architecture, and base and fine-tuned models are then converted alike."""
    fixes = []
    for name in ("config.json", "generation_config.json"):
        src, dst = os.path.join(base_dir, name), os.path.join(merged, name)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
            fixes.append(f"{name} copied from the base model into the merged model")
    return fixes


TOKENIZER_FILES = ("tokenizer.json", "preprocessor_config.json")  # faster-whisper reads these


def copy_files_arg(model_dir: str) -> str:
    """--copy_files for the tokenizer files the model folder actually has (run c2: the pinned
    base snapshot has no tokenizer.json, only vocab.json and merges.txt)."""
    names = [n for n in TOKENIZER_FILES if os.path.exists(os.path.join(model_dir, n))]
    return " --copy_files " + " ".join(names) if names else ""


def fill_missing_files(src_dir: str, dst_dir: str, names) -> list:
    """Copy files dst_dir lacks from src_dir. Used for ct2_base's tokenizer.json, taken from the
    fine-tuned export: a LoRA merge leaves the tokenizer unchanged (checked in run c2: same
    vocabulary and same token ids), so both models decode with the same tokenizer."""
    fixes = []
    for name in names:
        src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
            fixes.append(f"{name} copied from {os.path.basename(src_dir)} into {dst_dir}")
    return fixes


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    INFO["runtime"] = runtime_info()
    install()
    INFO["versions_after_install"] = versions()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration
    from unsloth import FastModel, is_bf16_supported

    INFO["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    INFO["cuda"] = torch.version.cuda
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
            output_dir=f"{WORK}/outputs",
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

    merged = f"{WORK}/merged"
    try:
        model.save_pretrained_merged(merged, tokenizer, save_method=None)
        INFO["merge_method"] = "unsloth save_pretrained_merged(save_method=None)"
    except Exception as exc:  # fall back to plain PEFT merge
        INFO["merge_unsloth_error"] = repr(exc)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged)
        tokenizer.save_pretrained(merged)
        INFO["merge_method"] = "peft merge_and_unload"
    INFO["env_fixes"] += fill_missing_config(merged, base_dir)
    q = CONFIG["runtime"]["ct2_quantization"]
    conv = "ct2-transformers-converter"
    for src, dst in ((merged, f"{OUT}/ct2_ft"), (base_dir, f"{OUT}/ct2_base")):
        sh(f"{conv} --model {src} --output_dir {dst} --quantization {q}{copy_files_arg(src)}")
    INFO["env_fixes"] += fill_missing_files(f"{OUT}/ct2_ft", f"{OUT}/ct2_base", TOKENIZER_FILES)

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
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
        INFO["status"] = "ok"
    except BaseException:
        INFO["status"] = "error"
        INFO["traceback"] = traceback.format_exc()
        code = 1
    os.makedirs(OUT, exist_ok=True)
    try:
        INFO["sha256"] = checksums(OUT)
    except Exception:
        INFO["sha256_error"] = traceback.format_exc()
    INFO["finished_unix"] = time.time()
    with open(f"{OUT}/run_info.json", "w", encoding="utf-8") as f:
        json.dump(INFO, f, indent=2, default=str)
    print(json.dumps({k: INFO.get(k) for k in ("gpu", "eval_loss_step0", "train_runtime_s")}))
    print(f"S3 DONE: status={INFO['status']}", flush=True)
    sys.exit(code)
