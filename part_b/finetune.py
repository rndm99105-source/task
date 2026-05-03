"""
Part B — Fine-tuning Whisper-small for Azerbaijani ASR
Train  : 200 samples from Common Voice 17.0 (az)
Val    : 50  samples
Test   : 100 samples

NOTE: Requires HuggingFace login + accepted Common Voice dataset terms.
"""

import os, torch, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jiwer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
warnings.filterwarnings("ignore")

from datasets import load_dataset, Audio
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from huggingface_hub import login

# ── HuggingFace Login ─────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️  HF_TOKEN not set. Dataset will fail without login.")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-small"
DATASET_NAME  = "mozilla-foundation/common_voice_17_0"
LANGUAGE      = "az"
LANGUAGE_FULL = "azerbaijani"
TASK          = "transcribe"

TRAIN_SAMPLES = 200
VAL_SAMPLES   = 50
TEST_SAMPLES  = 100

OUTPUT_DIR  = "./whisper-small-az"
RESULTS_DIR = "../results"
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
LR              = 1e-5
TRAIN_BATCH     = 8
EVAL_BATCH      = 8
MAX_STEPS       = 500
WARMUP_STEPS    = 50
EVAL_STEPS      = 100
SAVE_STEPS      = 100
GRAD_ACCUM      = 2

print(f"🎯  Fine-tuning {MODEL_NAME} → Azerbaijani ASR")
print("=" * 60)

# ── 1. DEVICE ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️   Device: {device}")
if device == "cpu":
    print("⚠️   No GPU found — use Colab GPU runtime for reasonable speed.")

# ── 2. DATASET ────────────────────────────────────────────────────────────────
print("\n📥 Loading dataset …")
def load_split(split_str):
    ds = load_dataset(DATASET_NAME, LANGUAGE, split=split_str)
    drop = [c for c in ["accent","age","client_id","down_votes","gender",
                         "locale","path","segment","up_votes","variant"]
            if c in ds.column_names]
    ds = ds.remove_columns(drop)
    return ds.cast_column("audio", Audio(sampling_rate=16_000))

train_raw = load_split(f"train[:{TRAIN_SAMPLES}]")
val_raw   = load_split(f"validation[:{VAL_SAMPLES}]")
test_raw  = load_split(f"test[:{TEST_SAMPLES}]")
print(f"  Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

# ── 3. MODEL & PROCESSOR ──────────────────────────────────────────────────────
print(f"\n🤖 Loading {MODEL_NAME} …")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE_FULL, task=TASK
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language              = LANGUAGE_FULL
model.generation_config.task                  = TASK
model.generation_config.forced_decoder_ids    = None
model = model.to(device)
print("✅  Model ready")

# ── 4. FEATURE PREPARATION ───────────────────────────────────────────────────
def prepare(batch):
    batch["input_features"] = processor.feature_extractor(
        batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"],
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("\n🔄 Extracting features …")
cols = ["audio", "sentence"]
train_ds = train_raw.map(prepare, remove_columns=train_raw.column_names)
val_ds   = val_raw.map(prepare,   remove_columns=val_raw.column_names)
test_ds  = test_raw.map(prepare,  remove_columns=test_raw.column_names)
print("✅  Features ready")

# ── 5. DATA COLLATOR ──────────────────────────────────────────────────────────
@dataclass
class SpeechCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        inp = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inp, return_tensors="pt")

        lbl = [{"input_ids": f["labels"]} for f in features]
        lbl_batch = self.processor.tokenizer.pad(lbl, return_tensors="pt")

        labels = lbl_batch["input_ids"].masked_fill(
            lbl_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

collator = SpeechCollator(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ── 6. METRIC ─────────────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# ── 7. TRAINING ARGUMENTS ─────────────────────────────────────────────────────
args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = TRAIN_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LR,
    warmup_steps                = WARMUP_STEPS,
    max_steps                   = MAX_STEPS,
    gradient_checkpointing      = True,
    fp16                        = torch.cuda.is_available(),
    eval_strategy               = "steps",
    per_device_eval_batch_size  = EVAL_BATCH,
    predict_with_generate       = True,
    generation_max_length       = 225,
    save_steps                  = SAVE_STEPS,
    eval_steps                  = EVAL_STEPS,
    logging_steps               = 25,
    report_to                   = ["none"],
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    push_to_hub                 = False,
)

# ── 8. TRAINER ────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    args                 = args,
    model                = model,
    train_dataset        = train_ds,
    eval_dataset         = val_ds,
    data_collator        = collator,
    compute_metrics      = compute_metrics,
    processing_class     = processor.feature_extractor,
    callbacks            = [EarlyStoppingCallback(early_stopping_patience=3)],
)

# ── 9. BASELINE EVAL ──────────────────────────────────────────────────────────
print("\n📊 Baseline evaluation (before fine-tuning) …")
baseline_eval = trainer.evaluate(test_ds)
baseline_wer  = baseline_eval.get("eval_wer", float("nan"))
print(f"  Baseline WER : {baseline_wer:.2f}%")

# ── 10. FINE-TUNE ─────────────────────────────────────────────────────────────
print(f"\n🚀 Fine-tuning … ({MAX_STEPS} steps)")
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅  Best model saved → {OUTPUT_DIR}")

# ── 11. FINE-TUNED EVAL ───────────────────────────────────────────────────────
print("\n📊 Fine-tuned evaluation …")
ft_eval  = trainer.evaluate(test_ds)
ft_wer   = ft_eval.get("eval_wer", float("nan"))
print(f"  Fine-tuned WER : {ft_wer:.2f}%")
print(f"  Δ WER          : {baseline_wer - ft_wer:+.2f}%  (improvement)")

# ── 12. COMPUTE CER FOR BOTH MODELS ──────────────────────────────────────────
cer_tx = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
    jiwer.Strip(), jiwer.ReduceToListOfListOfChars(),
])

def infer_all(m, proc, dataset_raw):
    m.eval()
    refs, hyps = [], []
    for sample in dataset_raw:
        ref = sample["sentence"].strip()
        feat = proc(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt",
        ).input_features.to(device)
        with torch.no_grad():
            ids = m.generate(feat, language=LANGUAGE_FULL, task=TASK, max_new_tokens=225)
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        refs.append(ref); hyps.append(hyp)
    return refs, hyps

print("\n🔄 Running final comparison inference …")
# Baseline (reload original weights)
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
base_proc  = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE_FULL, task=TASK)
base_refs, base_hyps = infer_all(base_model, base_proc, test_raw)

# Fine-tuned
ft_model  = WhisperForConditionalGeneration.from_pretrained(OUTPUT_DIR).to(device)
ft_proc   = WhisperProcessor.from_pretrained(OUTPUT_DIR)
ft_refs, ft_hyps = infer_all(ft_model, ft_proc, test_raw)

def calc_cer(refs, hyps):
    return jiwer.cer(refs, hyps,
                     reference_transform=cer_tx,
                     hypothesis_transform=cer_tx) * 100

base_cer = calc_cer(base_refs, base_hyps)
ft_cer   = calc_cer(ft_refs,   ft_hyps)

# ── 13. COMPARISON TABLE ──────────────────────────────────────────────────────
cmp = pd.DataFrame([
    {"Model": "Whisper-small (Baseline)",   "WER (%)": round(baseline_wer, 2), "CER (%)": round(base_cer, 2)},
    {"Model": "Whisper-small (Fine-tuned)", "WER (%)": round(ft_wer, 2),       "CER (%)": round(ft_cer, 2)},
])
cmp["ΔWER (%)"] = [0.0, round(baseline_wer - ft_wer, 2)]
cmp["ΔCER (%)"] = [0.0, round(base_cer - ft_cer,     2)]
print("\n📋 Comparison Table:")
print(cmp.to_string(index=False))
cmp.to_csv(f"{RESULTS_DIR}/part_b_comparison.csv", index=False)

# ── 14. TRAINING CURVES ───────────────────────────────────────────────────────
history = trainer.state.log_history
t_steps, t_loss, e_steps, e_wer, e_loss = [], [], [], [], []
for log in history:
    if "loss" in log and "eval_loss" not in log:
        t_steps.append(log["step"]); t_loss.append(log["loss"])
    if "eval_wer"  in log: e_wer.append(log["eval_wer"])
    if "eval_loss" in log:
        e_steps.append(log["step"]); e_loss.append(log["eval_loss"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(t_steps, t_loss,  label="Train Loss",  color="#4F8EF7", lw=2)
if e_loss:
    axes[0].plot(e_steps, e_loss, label="Val Loss", color="#E74C3C", lw=2, ls="--")
axes[0].set(xlabel="Steps", ylabel="Loss", title="Train & Validation Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

if e_wer:
    axes[1].plot(e_steps[:len(e_wer)], e_wer, color="#27AE60", lw=2, marker="o", ms=5)
    axes[1].set(xlabel="Steps", ylabel="WER (%)", title="Validation WER per Step")
    axes[1].grid(alpha=0.3)

plt.suptitle("Part B — Fine-tuning Whisper-small (Azerbaijani)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/part_b_training_curves.png", dpi=150)
plt.show()
print(f"\n📈  Training curves saved → {RESULTS_DIR}/part_b_training_curves.png")
print(f"💾  Comparison table  saved → {RESULTS_DIR}/part_b_comparison.csv")
