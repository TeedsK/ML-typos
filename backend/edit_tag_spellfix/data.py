# edit_tag_spellfix/data.py
"""
Dataset preparation for edit-tag spell-correction.
"""

from __future__ import annotations
import argparse, json, os, random, re # Added re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from .tags import build_tag_vocab, KEEP, TaggedExample, strip_prefix, is_replace, is_insert

# -----------------------------------------------------------
# Character whitelist for filtering (customize as needed)
# -----------------------------------------------------------
# Allows English letters (upper and lower), numbers, space, and common punctuation.
# Add any other characters you want to allow (e.g., accented characters if desired).
ALLOWED_CHAR_PATTERN = re.compile(r"^[a-zA-Z0-9 .,?!'\-_]+$") # Added hyphen and underscore

def is_token_clean(token: str) -> bool:
    """Checks if a token consists only of allowed characters."""
    if not token: # Allow empty tokens if they can occur
        return True
    return bool(ALLOWED_CHAR_PATTERN.match(token))

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def read_csv_with_tags(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["original_text", "edit_tags"])
    return df

# -----------------------------------------------------------
# Core class
# -----------------------------------------------------------
class TagDatasetBuilder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        seed: int = 42,
        filter_data: bool = True, # Add a flag to control filtering
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)
        self.filter_data = filter_data

        self.tag_vocab_built = False
        self.tag2id: dict[str, int] = {}

    def _is_example_clean(self, orig_tokens: List[str], tag_seq: List[str]) -> bool:
        """Checks if all original tokens and relevant tokens in tags are clean."""
        if not all(is_token_clean(token) for token in orig_tokens):
            return False
        
        for tag in tag_seq:
            if is_replace(tag) or is_insert(tag):
                token_to_check = strip_prefix(tag)
                if not is_token_clean(token_to_check):
                    return False
        return True

    def analyse(self, df: pd.DataFrame) -> List[Tuple[List[str], List[str]]]:
        examples = []
        tagged_objs_for_vocab = []
        
        processed_rows = 0
        filtered_rows = 0

        for index, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Processing and filtering rows",
        ):
            processed_rows += 1
            orig_text = str(row["original_text"])
            edit_tags_str = str(row["edit_tags"])

            orig_tokens = orig_text.strip().split()
            tag_seq = edit_tags_str.strip().split()

            if not orig_tokens and not tag_seq: # Allow empty examples if they make sense
                examples.append((orig_tokens, tag_seq))
                tagged_objs_for_vocab.append(TaggedExample(tokens=orig_tokens, tags=tag_seq, gap_tags=[]))
                continue

            if len(orig_tokens) != len(tag_seq):
                # print(f"Warning: Mismatch in token and tag sequence length for row {index}. Skipping.")
                filtered_rows +=1
                continue
            
            if self.filter_data and not self._is_example_clean(orig_tokens, tag_seq):
                filtered_rows += 1
                continue

            examples.append((orig_tokens, tag_seq))
            tagged_objs_for_vocab.append(TaggedExample(tokens=orig_tokens, tags=tag_seq, gap_tags=[]))
        
        print(f"Total rows processed: {processed_rows}")
        if self.filter_data:
            print(f"Rows filtered out due to character restrictions: {filtered_rows}")
        print(f"Rows remaining: {len(examples)}")

        if not examples:
            print("Warning: No examples remained after processing/filtering. Check your data and filter criteria.")
            # Still build an empty vocab if that's the case, or handle error
            self.tag2id = {KEEP: 0} # Basic vocab
            self.tag_vocab_built = True
            return []


        tag_list, self.tag2id = build_tag_vocab(tagged_objs_for_vocab)
        self.tag_vocab_built = True
        print(f"Tag vocab size after filtering: {len(tag_list)}")
        if tag_list:
            print(f"Sample tags from vocab: {random.sample(tag_list, min(len(tag_list), 5))}")
        return examples

    def encode_examples(self, examples):
        assert self.tag_vocab_built
        tag2id = self.tag2id
        tok = self.tokenizer
        
        # Ensure KEEP tag is in vocab, if not, add it.
        # This is critical if filtering heavily reduces the vocab.
        if KEEP not in tag2id:
            print(f"Warning: '{KEEP}' tag not found in vocabulary after filtering. Adding it.")
            if tag2id: # if vocab is not empty
                max_id = max(tag2id.values()) if tag2id else -1
                tag2id[KEEP] = max_id + 1
            else: # vocab is empty
                tag2id[KEEP] = 0

        default_tag_id = tag2id[KEEP]


        encoded_rows = []
        for orig_tokens, tag_seq in tqdm(examples, desc="Tokenising"):
            enc = tok(
                orig_tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
            word_ids = enc.word_ids()
            labels = [-100] * len(word_ids)

            for idx, w_id in enumerate(word_ids):
                if w_id is None:
                    continue
                if idx == 0 or word_ids[idx - 1] != w_id:
                    tag = tag_seq[w_id] if w_id < len(tag_seq) else KEEP
                    labels[idx] = tag2id.get(tag, default_tag_id) # Use default if tag somehow missing post-filtering

            encoded_rows.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": labels,
                }
            )
        return Dataset.from_list(encoded_rows)

    def build_dataset(
        self, df: pd.DataFrame, val_split: float | None = 0.05
    ) -> DatasetDict:
        examples = self.analyse(df)
        if not examples: # No examples after filtering
             # Return empty datasets
            return DatasetDict(train=Dataset.from_list([]), validation=Dataset.from_list([]))

        if val_split and 0 < val_split < 1.0:
            self.rng.shuffle(examples)
            n_val = int(len(examples) * val_split)
            if n_val == 0 and len(examples) > 1 : n_val = 1 # Ensure at least one val sample if possible
            if n_val >= len(examples): n_val = max(0, len(examples) -1)


            val_ex = examples[:n_val]
            train_ex = examples[n_val:]
            
            train_ds = self.encode_examples(train_ex) if train_ex else Dataset.from_list([])
            val_ds = self.encode_examples(val_ex) if val_ex else Dataset.from_list([])
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_filter_data", action="store_true", help="Disable filtering of non-English/special characters.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    print("Loading data …")
    df = read_csv_with_tags(args.csv)
    print(f"{len(df):,} rows read from CSV.")

    print("Initialising tokenizer:", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, add_prefix_space=True)

    should_filter = not args.no_filter_data
    if should_filter:
        print(f"Data filtering is ENABLED. Using pattern: {ALLOWED_CHAR_PATTERN.pattern}")
    else:
        print("Data filtering is DISABLED.")

    builder = TagDatasetBuilder(tok, max_length=args.max_length, seed=args.seed, filter_data=should_filter)
    dsdict = builder.build_dataset(df, val_split=args.val_split)

    if not dsdict["train"] and not (dsdict.get("validation") and dsdict["validation"]):
        print("Critical: Both training and validation datasets are empty. Aborting.")
        return

    print("Saving HF dataset to:", args.out)
    dsdict.save_to_disk(args.out)

    tag_json = Path(args.out) / "tag2id.json"
    with tag_json.open("w", encoding="utf8") as f:
        json.dump(builder.tag2id, f, ensure_ascii=False, indent=2)
    print("Tag vocab written to", tag_json)


if __name__ == "__main__":
    main()