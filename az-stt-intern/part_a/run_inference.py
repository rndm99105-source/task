#!/usr/bin/env python3
"""
Part A – Run inference and compute WER/CER on the Azerbaijani Common Voice dataset.

Usage:
    python run_inference.py \
        --dataset_name mozilla-foundation/common_voice_17_0 \
        --language az \
        --model_name valiyevfagan/whisper-small-az \
        --max_samples 200

This script loads the specified ASR model and dataset, runs inference on the test
split, and computes the average word error rate (WER) and character error rate
(CER). Results (per‑sample predictions and references) are saved as a CSV file
in the `results/` directory.
"""
import argparse
import os

import pandas as pd
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR inference and evaluate WER/CER")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on the Hugging Face Hub")
    parser.add_argument("--language", type=str, default="az", help="Language code (e.g., 'az' for Azerbaijani)")
    parser.add_argument("--model_name", type=str, required=True, help="Model checkpoint to use for inference")
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum number of test samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="../results", help="Directory to save the results CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset {args.dataset_name}…")
    dataset = load_dataset(args.dataset_name, args.language, split="test")
    if args.max_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    # Load model and processor
    print(f"Loading model {args.model_name}…")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    references = []
    predictions = []

    for idx, example in enumerate(dataset):
        audio = example["audio"]
        # Whisper processor expects 16 kHz float array
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Normalise both reference and prediction (lowercase, remove punctuation)
        reference = example["sentence"]
        references.append(reference.lower())
        predictions.append(transcription.lower())

        print(f"{idx+1}/{len(dataset)}: ref='{reference[:50]}…', pred='{transcription[:50]}…'")

    # Compute metrics
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print(f"Average WER: {wer:.4f}")
    print(f"Average CER: {cer:.4f}")

    # Save results to CSV
    df = pd.DataFrame({
        "reference": references,
        "prediction": predictions
    })
    csv_path = os.path.join(args.output_dir, "part_a_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
