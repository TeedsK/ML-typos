# edit_tag_spellfix/predict_verbose.py
"""
Verbose inference: returns per-token tag plus tag-probability vector.

Example
-------
python -m edit_tag_spellfix.predict_verbose \
        --model_dir ".\\models\\roberta_tag_82k_v2" \
        --sentence "a machine learning and fuil stack engineer huilding web tools and apps that deliver measurabie impacts" \
        --top_k 3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .model import RobertaTagger
from .tags  import KEEP, DELETE, strip_prefix, is_replace

# -----------------------------------------------------------------------------
def predict_verbose(
    sent: str,
    model: RobertaTagger,
    tokenizer,
    id2tag: Dict[int, str],
    device: str = "cpu",
    top_k: int | None = None,
) -> Dict:
    """Return dict with tokens, tags, probs (top-k or full)."""
    tokens = sent.strip().split()
    batch = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    word_ids = batch.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**enc)["logits"]          # [1, seq_len, num_tags]
    probs = F.softmax(logits, dim=-1).squeeze(0) # [seq_len, num_tags]

    results = []
    for idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        # first sub-token of each word
        if idx == 0 or word_ids[idx-1] != w_id:
            token_probs = probs[idx]
            tag_id      = int(torch.argmax(token_probs))
            tag_label   = id2tag[tag_id]

            if top_k:
                topk_probs, topk_ids = torch.topk(token_probs, k=top_k)
                dist = {
                    id2tag[int(i)]: float(p)
                    for i, p in zip(topk_ids, topk_probs)
                }
            else:
                # full dist → be cautious, could be large
                dist = {id2tag[i]: float(p) for i, p in enumerate(token_probs)}

            results.append({
                "token": tokens[w_id],
                "tag":   tag_label,
                "probs": dist,
            })

    # reconstruct corrected sentence for convenience
    corrected_tokens = []
    for orig, item in zip(tokens, results):
        tg = item["tag"]
        if tg == KEEP:
            corrected_tokens.append(orig)
        elif tg == DELETE:
            continue
        elif is_replace(tg):
            corrected_tokens.append(strip_prefix(tg))
        else:
            corrected_tokens.append(orig)   # fall-back

    return {
        "original":  sent,
        "corrected": " ".join(corrected_tokens),
        "tokens":    results,
    }

# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--sentence", help="Input sentence")
    ap.add_argument("--input_file", help="File with one sentence per line")
    ap.add_argument("--top_k", type=int, default=5,
                    help="How many highest-prob tags to keep (None = full dist)")
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
                freeze_encoder=True)

    if args.sentence:
        print(json.dumps(
            predict_verbose(args.sentence, model, tok, id2tag, device, args.top_k),
            indent=2, ensure_ascii=False
        ))
    else:
        for line in open(args.input_file, encoding="utf8"):
            print(json.dumps(
                predict_verbose(line.rstrip("\n"), model, tok, id2tag, device, args.top_k),
                ensure_ascii=False
            ))

if __name__ == "__main__":
    main()
