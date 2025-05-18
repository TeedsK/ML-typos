# edit_tag_spellfix/data.py
"""
Dataset preparation for edit-tag spell-correction.
--------------------------------------------------

Usage (creates HF arrow cache + tag vocab JSON):

    python -m edit_tag_spellfix.data \
        --csv  path/to/developer_typo_dataset_60k.csv \
        --out  data/60k_roberta \
        --tokenizer roberta-base

This will write
   • data/60k_roberta/train.arrow  (and /validation.arrow if --split 0.x)
   • data/60k_roberta/tag2id.json
"""

from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import textdistance
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from .tags import diff_to_tags, build_tag_vocab, KEEP, DELETE

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def minimal_edit(orig: str, corr: str, threshold: float = 0.30) -> bool:
    """Return True if Levenshtein distance ≤ threshold of max(len))."""
    dist = textdistance.levenshtein.distance(orig.lower(), corr.lower())
    mx = max(len(orig), len(corr))
    return (dist / mx) <= threshold


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["original_text", "corrected_text"])
    # keep minimal-edit pairs only
    keep_mask = df.apply(
        lambda r: minimal_edit(r["original_text"], r["corrected_text"]), axis=1
    )
    return df[keep_mask]


# -----------------------------------------------------------
# Core class
# -----------------------------------------------------------
class TagDatasetBuilder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)

        self.tag_vocab_built = False
        self.tag2id: dict[str, int] = {}

    # -----------------------
    # Step 1: analyse corpus
    # -----------------------
    def analyse(self, df: pd.DataFrame) -> List[Tuple[List[str], List[str]]]:
        examples = []
        tagged_objs = []
        for orig, corr in tqdm(
            zip(df["original_text"], df["corrected_text"]),
            total=len(df),
            desc="Diff-tagging",
        ):
            orig_tokens = orig.strip().split()
            corr_tokens = corr.strip().split()
            te = diff_to_tags(orig_tokens, corr_tokens)
            examples.append((orig_tokens, te.tags))  # ignore gap_tags for now
            tagged_objs.append(te)
        # build vocab
        tag_list, self.tag2id = build_tag_vocab(tagged_objs)
        self.tag_vocab_built = True
        print(f"Tag vocab size: {len(tag_list)}")
        return examples

    # -----------------------
    # Step 2: encode to HF
    # -----------------------
    def encode_examples(self, examples):
        """Return a HuggingFace Dataset list of dicts."""
        assert self.tag_vocab_built
        tag2id = self.tag2id
        tok = self.tokenizer

        encoded_rows = []
        for orig_tokens, tag_seq in tqdm(examples, desc="Tokenising"):
            enc = tok(
                orig_tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
            word_ids = enc.word_ids()  # list len = seq_len
            labels = [-100] * len(word_ids)

            for idx, w_id in enumerate(word_ids):
                if w_id is None:
                    continue
                # assign tag only to *first* sub-token of each word
                if idx == 0 or word_ids[idx - 1] != w_id:
                    tag = tag_seq[w_id] if w_id < len(tag_seq) else KEEP
                    labels[idx] = tag2id.get(tag, tag2id[KEEP])
                # subsequent sub-tokens keep -100

            encoded_rows.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": labels,
                }
            )
        return Dataset.from_list(encoded_rows)

    # -----------------------
    # Public one-shot method
    # -----------------------
    def build_dataset(
        self, df: pd.DataFrame, val_split: float | None = 0.05
    ) -> DatasetDict:
        examples = self.analyse(df)
        if val_split and 0 < val_split < 1.0:
            self.rng.shuffle(examples)
            n_val = int(len(examples) * val_split)
            val_ex = examples[:n_val]
            train_ex = examples[n_val:]
            train_ds = self.encode_examples(train_ex)
            val_ds = self.encode_examples(val_ex)
            return DatasetDict(train=train_ds, validation=val_ds)
        else:
            ds = self.encode_examples(examples)
            return DatasetDict(train=ds)


# -----------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--out", required=True, help="Output dir for HF dataset")
    ap.add_argument("--tokenizer", default="roberta-base")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--val_split", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data …")
    df = read_csv(args.csv)
    print(f"{len(df):,} minimal-edit pairs retained.")

    print("Initialising tokenizer:", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, add_prefix_space=True)

    builder = TagDatasetBuilder(tok, max_length=args.max_length)
    dsdict = builder.build_dataset(df, val_split=args.val_split)

    print("Saving HF dataset to:", args.out)
    dsdict.save_to_disk(args.out)

    # save tag vocab
    tag_json = Path(args.out) / "tag2id.json"
    with tag_json.open("w", encoding="utf8") as f:
        json.dump(builder.tag2id, f, ensure_ascii=False, indent=2)
    print("Tag vocab written to", tag_json)


if __name__ == "__main__":
    main()
