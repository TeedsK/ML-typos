# edit_tag_spellfix/train.py
"""
Train the RoBERTa edit-tag model.

Example:
    python -m edit_tag_spellfix.train \
        --data_dir  data/60k_roberta \
        --output_dir models/roberta_tag_60k \
        --epochs 3 \
        --batch_size 16 \
        --lr 2e-5
"""

from __future__ import annotations
import argparse, json, os, math
from pathlib import Path

import numpy as np
import evaluate
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from .model import RobertaTagger


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def compute_metrics(eval_pred, id2tag):
    """Simple token-level accuracy (ignoring label == -100)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    correct = (preds == labels) & mask
    acc = correct.sum() / mask.sum()

    return {"token_accuracy": acc}


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path created by data.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Load dataset + tag-vocab
    dsdict = load_from_disk(args.data_dir)
    tag_json = Path(args.data_dir) / "tag2id.json"
    tag2id = json.load(open(tag_json))
    id2tag = {v: k for k, v in tag2id.items()}
    num_tags = len(tag2id)
    print("Dataset:", dsdict)
    print("Tag vocab size:", num_tags)

    # 2. Tokenizer (same params as data prep)
    tok = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)

    # 3. Model
    model = RobertaTagger.from_pretrained_with_tags(
        args.base_model,          # positional – path to previous checkpoint
        tag_json,                 # tag vocab for v2 dataset
        freeze_encoder=True,      # keeps training under ~5 min on CPU
    )

    # 4. Data collator
    data_collator = DataCollatorForTokenClassification(tok, pad_to_multiple_of=8)

    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsdict["train"],
        eval_dataset=dsdict["validation"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2tag),
    )

    # 7. Train
    trainer.train()

    # 8. Save final artefacts
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)


if __name__ == "__main__":
    main()
