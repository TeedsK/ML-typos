# edit_tag_spellfix/train.py
"""
Train the RoBERTa edit-tag model.
"""

from __future__ import annotations
import argparse, json, os, re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import PredictionOutput

from .model import RobertaTagger


# -------------------------------------------------------------------
# Custom Trainer to handle large vocabulary evaluation
# -------------------------------------------------------------------
class CustomTrainer(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor | torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        model.eval()

        with torch.no_grad():
            loss, model_outputs = self.compute_loss(model, inputs, return_outputs=True)
            logits = model_outputs.get("logits") if isinstance(model_outputs, dict) else model_outputs[0]

        if prediction_loss_only or logits is None:
            return (loss, None, None)

        labels = inputs.get("labels")
        predictions = torch.argmax(logits, dim=-1)
        
        return (loss, predictions, labels)


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def compute_metrics(eval_pred: PredictionOutput, id2tag=None):
    predictions_indices, label_ids = eval_pred
    
    if predictions_indices is None or label_ids is None:
        return {}

    mask = label_ids != -100
    correct = (predictions_indices == label_ids) & mask
    acc = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0

    return {"token_accuracy": acc}


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path created by data.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unfreeze_encoder", action='store_true')
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--subsample_eval", type=int, default=0, help="Number of samples for evaluation (0=all).")
    
    # --- ADDED ARGUMENTS FOR FLEXIBILITY & TESTING ---
    ap.add_argument("--max_steps", type=int, default=-1, help="If set > 0, overrides num_epochs.")
    ap.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dsdict = load_from_disk(args.data_dir)
    tag_json = Path(args.data_dir) / "tag2id.json"
    try:
        with open(tag_json, 'r', encoding='utf-8') as f:
            tag2id = json.load(f)
    except Exception as e:
        print(f"Error loading tag2id.json: {e}")
        return
        
    id2tag = {v: k for k, v in tag2id.items()}
    num_tags = len(tag2id)
    print("Dataset:", dsdict)
    print("Tag vocab size:", num_tags)

    if "train" not in dsdict or not dsdict["train"]:
        print("Training dataset is missing or empty.")
        return

    eval_dataset = dsdict.get("validation")
    if eval_dataset:
        if args.subsample_eval > 0 and args.subsample_eval < len(eval_dataset):
            print(f"Subsampling validation set to {args.subsample_eval} samples.")
            eval_dataset = eval_dataset.select(range(args.subsample_eval))
    else:
        print("Validation dataset not found. Evaluation will be disabled.")
        args.evaluation_strategy = "no"

    tok = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)
    model = RobertaTagger.from_pretrained_with_tags(
        args.base_model,
        tag_json,
        freeze_encoder=(not args.unfreeze_encoder),
        dropout=args.dropout
    )
    print(f"Initializing model. Encoder frozen: {not args.unfreeze_encoder}")

    data_collator = DataCollatorForTokenClassification(tok, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Directly use the parsed arguments
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.evaluation_strategy if eval_dataset else "no",
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True if (eval_dataset and args.evaluation_strategy != "no") else False,
        metric_for_best_model="token_accuracy" if (eval_dataset and args.evaluation_strategy != "no") else None,
        greater_is_better=True if (eval_dataset and args.evaluation_strategy != "no") else None,
        seed=args.seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dsdict["train"],
        eval_dataset=eval_dataset,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete. Saving final model and tokenizer.")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / "tag2id.json", "w", encoding="utf8") as f:
        json.dump(tag2id, f, ensure_ascii=False, indent=2)
    print("Final model, tokenizer, and tag vocabulary saved to", args.output_dir)


if __name__ == "__main__":
    main()