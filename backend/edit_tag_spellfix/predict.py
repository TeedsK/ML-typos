# edit_tag_spellfix/predict.py
"""
Inference script for the edit-tag spell-correction model.

Examples
--------
# single sentence
python -m edit_tag_spellfix.predict \
        --model_dir ".\\models\\roberta_tag_60k_v2" \
        --sentence "a machine learning and fuil stack engineer huilding web tools and apps that deliver measurabie impacts"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from .model import RobertaTagger            # ← fixed import
from .tags  import KEEP, DELETE, is_replace, strip_prefix

# --------------------------------------------------------------------
def apply_tags(tokens: List[str], tags: List[str]) -> str:
    """Re-assemble corrected sentence from token/tag sequence."""
    out: List[str] = []
    for tok, tg in zip(tokens, tags):
        if tg == KEEP:
            out.append(tok)
        elif tg == DELETE:
            continue
        elif is_replace(tg):
            out.append(strip_prefix(tg))
        else:
            out.append(tok)    # fall-back
    return " ".join(out)

# --------------------------------------------------------------------
def predict_sentence(
    sent: str,
    model: RobertaTagger,
    tokenizer,
    id2tag: dict[int, str],
    device: str = "cpu",
) -> str:
    orig_tokens = sent.strip().split()
    batch = tokenizer(
        orig_tokens,
        is_split_into_words=True,
        return_tensors="pt",
    )
    word_ids = batch.word_ids(batch_index=0)  
    enc = {k: v.to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**enc)["logits"]
    pred_ids = logits.argmax(-1).squeeze().tolist()

    tags: List[str] = []
    for idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if idx == 0 or word_ids[idx - 1] != w_id:      # first sub-token
            tags.append(id2tag[pred_ids[idx]])

    return apply_tags(orig_tokens, tags)

# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--sentence",   help="Single sentence to correct")
    ap.add_argument("--input_file", help="File with one sentence per line")
    args = ap.parse_args()

    if not (args.sentence or args.input_file):
        sys.exit("Provide --sentence or --input_file")

    tok = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)
    tag2id = json.load(open(Path(args.model_dir) / "tag2id.json"))
    id2tag = {v: k for k, v in tag2id.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = RobertaTagger.from_pretrained_with_tags(
        args.model_dir,
        Path(args.model_dir) / "tag2id.json",
        freeze_encoder=True,       # encoder already frozen; fine
    ).to(device)

    if args.sentence:
        print(predict_sentence(args.sentence, model, tok, id2tag, device))
    else:
        with open(args.input_file, encoding="utf8") as f:
            for line in f:
                print(predict_sentence(line.rstrip("\n"), model, tok, id2tag, device))

if __name__ == "__main__":
    main()
