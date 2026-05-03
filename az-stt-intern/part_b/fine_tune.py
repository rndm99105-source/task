#!/usr/bin/env python3
"""
Part B – Fine‑tune Whisper model on a small subset of the Azerbaijani dataset.

Usage example:
    python fine_tune.py \
        --dataset_name mozilla-foundation/common_voice_17_0 \
        --language az \
        --model_name valiyevfagan/whisper-small-az \
        --output_dir ../results/fine_tuned_model \
        --num_epochs 3 \
        --learning_rate 1e-5

This script performs sequence-to-sequence fine‑tuning using Hugging Face's
Trainer API. It monitors validation WER after each epoch and saves the best
checkpoint.
"""
import argparse
import os

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Fine‑tune Whisper model on Azerbaijani dataset")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--language", type=str, default="az")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    return parser.parse_args()


@dataclass
class DataCollatorWhisper:
    """Data collator that pads input features and labels for Whisper fine‑tuning."""
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [f["input_features"] for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            padding=True,
            return_tensors="pt"
        )
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    # Load audio to float32 array and compute log-Mel features
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
    # Tokenize the target text
    batch["labels"] = processor.tokenizer(batch["sentence"], return_tensors="pt").input_ids[0]
    return batch


def compute_metrics_fn(processor, wer_metric):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 with pad token id
        preds = torch.argmax(torch.tensor(preds), dim=-1)
        label_ids = labels
        # Decode
        pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=[p.lower() for p in pred_str], references=[l.lower() for l in label_str])
        return {"wer": wer}
    return compute_metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_name, args.language)
    train_ds = dataset["train"]
    eval_ds = dataset["validation"] if "validation" in dataset else dataset["test"]
    if args.max_samples:
        train_ds = train_ds.select(range(min(len(train_ds), args.max_samples)))
        eval_ds = eval_ds.select(range(min(len(eval_ds), max(1, args.max_samples // 10))))

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Freeze the encoder to speed up fine‑tuning and reduce overfitting
    model.freeze_encoder()

    # Preprocess datasets
    train_ds = train_ds.map(lambda b: prepare_dataset(b, processor), remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(lambda b: prepare_dataset(b, processor), remove_columns=eval_ds.column_names)

    data_collator = DataCollatorWhisper(processor)
    wer_metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_batch_size // 2),
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics_fn(processor, wer_metric),
    )

    trainer.train()
    # Save the best model
    trainer.save_model(args.output_dir)
    print(f"Fine‑tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
