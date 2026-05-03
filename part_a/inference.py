"""
Part A — ASR Baseline Inference
Model  : openai/whisper-small
Dataset: Google FLEURS — Azerbaijani (az_az)
         [Replaces Common Voice 17 which moved to Mozilla Data Collective in Oct 2025]
Metrics: WER, CER
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jiwer
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-small"
DATASET_NAME  = "google/fleurs"
DATASET_CONFIG = "az_az"          # Azerbaijani
TEXT_COLUMN   = "transcription"   # FLEURS uses 'transcription' not 'sentence'
LANGUAGE_FULL = "azerbaijani"
TASK          = "transcribe"
MAX_SAMPLES   = 200
RESULTS_DIR   = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"🔊  ASR Baseline — {MODEL_NAME}")
print(f"📦  Dataset      — {DATASET_NAME} [{DATASET_CONFIG}]")
print("=" * 60)

# ── 1. DEVICE ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️   Device: {device}")

# ── 2. DATASET ────────────────────────────────────────────────────────────────
print("\n📥 Loading dataset …")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", trust_remote_code=True)
if len(dataset) > MAX_SAMPLES:
    dataset = dataset.select(range(MAX_SAMPLES))
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
print(f"✅  {len(dataset)} test samples loaded")

# ── 3. MODEL ──────────────────────────────────────────────────────────────────
print(f"\n🤖 Loading {MODEL_NAME} …")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE_FULL, task=TASK
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("✅  Model ready")

# ── 4. INFERENCE ──────────────────────────────────────────────────────────────
def transcribe(sample: dict) -> str:
    features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(
            features,
            language=LANGUAGE_FULL,
            task=TASK,
            max_new_tokens=225,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

print("\n🔄 Running inference …")
references, hypotheses = [], []
for i, sample in enumerate(dataset):
    references.append(sample[TEXT_COLUMN].strip())
    hypotheses.append(transcribe(sample).strip())
    if (i + 1) % 20 == 0:
        print(f"   {i+1}/{len(dataset)}")

print(f"✅  Done — {len(hypotheses)} samples processed")

# ── 5. METRICS ────────────────────────────────────────────────────────────────
wer_tx = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(), jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])
cer_tx = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
    jiwer.Strip(), jiwer.ReduceToListOfListOfChars(),
])

overall_wer = jiwer.wer(references, hypotheses,
                         reference_transform=wer_tx,
                         hypothesis_transform=wer_tx)
overall_cer = jiwer.cer(references, hypotheses,
                         reference_transform=cer_tx,
                         hypothesis_transform=cer_tx)

print(f"\n{'='*60}")
print(f"  📌  Average WER : {overall_wer*100:.2f}%")
print(f"  📌  Average CER : {overall_cer*100:.2f}%")
print(f"{'='*60}")

# ── 6. PER-SAMPLE SCORES ──────────────────────────────────────────────────────
sample_wers, sample_cers = [], []
for ref, hyp in zip(references, hypotheses):
    try:   w = jiwer.wer(ref.lower(), hyp.lower())
    except: w = 1.0
    try:   c = jiwer.cer(ref.lower(), hyp.lower())
    except: c = 1.0
    sample_wers.append(w)
    sample_cers.append(c)

df = pd.DataFrame({
    "reference" : references,
    "hypothesis": hypotheses,
    "WER"       : sample_wers,
    "CER"       : sample_cers,
})

# ── 7. BEST & WORST 5 ─────────────────────────────────────────────────────────
def show_samples(label, rows):
    print(f"\n{label}")
    for _, r in rows.iterrows():
        print(f"  WER={r['WER']*100:5.1f}%  REF: {r['reference'][:70]}")
        print(f"                   HYP: {r['hypothesis'][:70]}")

show_samples("🏆  Top-5 Best  (lowest WER):", df.nsmallest(5, "WER"))
show_samples("❌  Top-5 Worst (highest WER):", df.nlargest(5, "WER"))

# ── 8. SAVE RESULTS ───────────────────────────────────────────────────────────
df.to_csv(f"{RESULTS_DIR}/part_a_results.csv", index=False)

summary = pd.DataFrame([{
    "Model"   : MODEL_NAME,
    "Dataset" : f"{DATASET_NAME} [{DATASET_CONFIG}]",
    "Samples" : len(df),
    "WER (%)" : round(overall_wer * 100, 2),
    "CER (%)" : round(overall_cer * 100, 2),
}])
summary.to_csv(f"{RESULTS_DIR}/part_a_summary.csv", index=False)
print(f"\n💾  Results saved → {RESULTS_DIR}/part_a_results.csv")

# ── 9. PLOT WER DISTRIBUTION ──────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.hist(df["WER"], bins=20, color="#4F8EF7", edgecolor="white", alpha=0.85)
plt.axvline(overall_wer, color="#E74C3C", linestyle="--", linewidth=2,
            label=f"Mean WER = {overall_wer*100:.1f}%")
plt.xlabel("WER (per sample)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Part A — WER Distribution  |  Baseline: Whisper-small (az)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/part_a_wer_distribution.png", dpi=150)
plt.show()
print("📈  WER distribution chart saved.")
