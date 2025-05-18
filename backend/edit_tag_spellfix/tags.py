# edit_tag_spellfix/tags.py
"""
Tag-scheme utilities for token-level spell-correction.
Now handles simple delete+insert cases as a single REPLACE tag.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import difflib

# ---------------------------------------------------------------------
# Tag constants
# ---------------------------------------------------------------------
KEEP    = "KEEP"
DELETE  = "DELETE"
INSERT  = "INSERT_"      # INSERT_token
REPLACE = "REPLACE_"     # REPLACE_token

def is_insert(tag: str)  -> bool: return tag.startswith(INSERT)
def is_replace(tag: str) -> bool: return tag.startswith(REPLACE)
def strip_prefix(tag: str) -> str:
    return tag.split("_", 1)[1] if "_" in tag else ""

# ---------------------------------------------------------------------
# Dataclass for a tagged example
# ---------------------------------------------------------------------
@dataclass
class TaggedExample:
    tokens:   List[str]  # original tokens
    tags:     List[str]  # one per original token
    gap_tags: List[str]  # len = len(tokens)+1   (kept for completeness)

# ---------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------
def diff_to_tags(orig_tokens: List[str], cor_tokens: List[str]) -> TaggedExample:
    """
    Align two token lists and convert edits into KEEP / DELETE / REPLACE_xxx
    plus optional INSERT_xxx gap tags.
    """
    matcher = difflib.SequenceMatcher(None, orig_tokens, cor_tokens)
    tags: List[str] = []
    gaps: List[str] = ["KEEP_GAP"] * (len(orig_tokens) + 1)

    opcodes = matcher.get_opcodes()
    i = 0
    while i < len(opcodes):
        op, i1, i2, j1, j2 = opcodes[i]

        # ------- equal -------
        if op == "equal":
            tags.extend([KEEP] * (i2 - i1))

        # ------- delete (maybe replace) -------
        elif op == "delete":
            # Look-ahead: single-token insert immediately after?
            if (i + 1 < len(opcodes)
                and opcodes[i + 1][0] == "insert"
                and (i2 - i1) == 1                # one token deleted
                and (opcodes[i + 1][4] - opcodes[i + 1][3]) == 1):  # one inserted
                # fuse into REPLACE_x
                new_tok = cor_tokens[opcodes[i + 1][3]]
                tags.append(f"{REPLACE}{new_tok}")
                i += 1        # skip the following insert opcode
            else:
                tags.extend([DELETE] * (i2 - i1))

        # ------- replace (lengths equal) -------
        elif op == "replace":
            span_orig = orig_tokens[i1:i2]
            span_cor  = cor_tokens[j1:j2]
            if len(span_orig) == len(span_cor):
                for new_tok in span_cor:
                    tags.append(f"{REPLACE}{new_tok}")
            else:
                # fall back: delete + insert at gap
                tags.extend([DELETE] * len(span_orig))
                gap_idx = len(tags)               # after deletes
                gaps[gap_idx] = " ".join(f"{INSERT}{t}" for t in span_cor)

        # ------- insert -------
        elif op == "insert":
            gap_idx = len(tags)
            gaps[gap_idx] = " ".join(f"{INSERT}{t}" for t in cor_tokens[j1:j2])

        i += 1

    assert len(tags) == len(orig_tokens)
    return TaggedExample(orig_tokens, tags, gaps)

# ---------------------------------------------------------------------
# Tag vocabulary
# ---------------------------------------------------------------------
def build_tag_vocab(examples: List[TaggedExample]):
    tag_set = {KEEP, DELETE}
    for ex in examples:
        tag_set.update(ex.tags)
    tag_list = sorted(tag_set)
    return tag_list, {t: i for i, t in enumerate(tag_list)}

# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    orig = "a fuil stack engineer huilding web apps".split()
    cor  = "a full-stack engineer building web apps".split()
    te = diff_to_tags(orig, cor)
    print("TOKENS :", te.tokens)
    print("TAGS   :", te.tags)
