# edit_tag_spellfix/hybrid_predict.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer
from symspellpy import SymSpell, Verbosity # Import SymSpell

from .model import RobertaTagger
from .tags import KEEP, DELETE, is_replace, strip_prefix

def hybrid_predict(
    sentence: str,
    roberta_model: RobertaTagger,
    tokenizer: AutoTokenizer,
    id2tag: dict[int, str],
    sym_spell: SymSpell,
    device: str = "cpu",
    max_symspell_edit_distance: int = 2,
) -> str:
    original_tokens = sentence.strip().split()
    if not original_tokens:
        return ""

    batch = tokenizer(
        original_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 512
    )
    word_ids = batch.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in batch.items()}

    roberta_model.eval()
    with torch.no_grad():
        outputs = roberta_model(**enc)
        logits = outputs.get("logits")
        
        if logits is None:
            print("Error: Could not get logits from RoBERTa model.", file=sys.stderr)
            return sentence 

        roberta_pred_ids_tensor = torch.argmax(logits, dim=-1).squeeze(0)
        
        if roberta_pred_ids_tensor.ndim == 0:
            roberta_pred_ids_list = [roberta_pred_ids_tensor.item()]
        else:
            roberta_pred_ids_list = roberta_pred_ids_tensor.tolist()

    roberta_tags = [KEEP] * len(original_tokens)
    current_word_idx = -1
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != current_word_idx:
            current_word_idx = word_id
            if token_idx < len(roberta_pred_ids_list) and word_id < len(original_tokens):
                 tag_id = roberta_pred_ids_list[token_idx]
                 roberta_tags[word_id] = id2tag.get(tag_id, KEEP)

    # --- DEBUGGING OUTPUT ---
    # print("\n--- RoBERTa Debug ---")
    # for i, token in enumerate(original_tokens):
    #     tag_to_print = roberta_tags[i] if i < len(roberta_tags) else "N/A (Alignment Issue)"
    #     print(f"Token: '{token}', RoBERTa_Tag: '{tag_to_print}'")
    # print("---------------------\n")

    corrected_tokens: list[str] = []
    for i, token in enumerate(original_tokens):
        roberta_tag = roberta_tags[i] if i < len(roberta_tags) else KEEP

        consult_symspell = False
        if roberta_tag == DELETE:
            continue

        if roberta_tag == KEEP:
            lookup_results = sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=0, include_unknown=False)
            if not lookup_results:
                consult_symspell = True
            else:
                corrected_tokens.append(token)
        else: 
            consult_symspell = True

        if consult_symspell:
            suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=max_symspell_edit_distance, include_unknown=True)
            if suggestions:
                corrected_tokens.append(suggestions[0].term)
            else:
                corrected_tokens.append(token)
                
    return " ".join(corrected_tokens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory of the trained RoBERTa model")
    ap.add_argument("--base_dictionary_path", required=True, help="Path to SymSpell base frequency dictionary (e.g., general English)")
    ap.add_argument("--custom_dictionary_path", help="Optional path to a custom domain-specific frequency dictionary")
    ap.add_argument("--sentence", help="Single sentence to correct")
    ap.add_argument("--input_file", help="File with one sentence per line")
    ap.add_argument("--max_edit_distance", type=int, default=2, help="SymSpell max edit distance for lookup")
    ap.add_argument("--prefix_length", type=int, default=7, help="SymSpell prefix length")
    args = ap.parse_args()

    if not (args.sentence or args.input_file):
        sys.exit("Provide --sentence or --input_file")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    roberta_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)
    tag2id_path = Path(args.model_dir) / "tag2id.json"
    if not tag2id_path.exists():
        sys.exit(f"Error: tag2id.json not found in {args.model_dir}")
    with open(tag2id_path, 'r', encoding='utf-8') as f:
        tag2id = json.load(f)
    id2tag = {v: k for k, v in tag2id.items()}

    roberta_model = RobertaTagger.from_pretrained_with_tags(
        args.model_dir,
        tag2id_path,
        freeze_encoder=True,
    ).to(device)

    # Load SymSpell with combined dictionaries
    sym_spell = SymSpell(max_dictionary_edit_distance=args.max_edit_distance, prefix_length=args.prefix_length)
    
    print(f"Loading base dictionary from: {args.base_dictionary_path}")
    if not sym_spell.load_dictionary(args.base_dictionary_path, term_index=0, count_index=1, encoding="utf-8"):
        print(f"Error: Base SymSpell dictionary not loaded from {args.base_dictionary_path}.", file=sys.stderr)
        return # Exit if base dictionary fails to load
    print(f"SymSpell dictionary: {sym_spell.word_count} words after loading base.")

    if args.custom_dictionary_path:
        custom_path = Path(args.custom_dictionary_path)
        if custom_path.exists():
            print(f"Loading custom dictionary from: {args.custom_dictionary_path}")
            # SymSpell's load_dictionary can be called multiple times; it merges the dictionaries.
            # Terms from newer dictionaries update frequencies if they already exist.
            if not sym_spell.load_dictionary(str(custom_path), term_index=0, count_index=1, encoding="utf-8"):
                print(f"Warning: Custom SymSpell dictionary at {args.custom_dictionary_path} could not be fully loaded or is empty.", file=sys.stderr)
            print(f"SymSpell dictionary: {sym_spell.word_count} words after loading custom.")
        else:
            print(f"Warning: Custom dictionary path specified but not found: {args.custom_dictionary_path}", file=sys.stderr)
    
    if sym_spell.word_count == 0:
        print("Error: SymSpell dictionary is empty after attempting to load all specified files. Cannot proceed.", file=sys.stderr)
        return

    if args.sentence:
        corrected = hybrid_predict(args.sentence, roberta_model, roberta_tokenizer, id2tag, sym_spell, device, args.max_edit_distance)
        print(corrected)
    else:
        with open(args.input_file, encoding="utf8") as f:
            for line in f:
                stripped_line = line.rstrip("\n")
                if stripped_line:
                    corrected = hybrid_predict(stripped_line, roberta_model, roberta_tokenizer, id2tag, sym_spell, device, args.max_edit_distance)
                    print(corrected)

if __name__ == "__main__":
    main()