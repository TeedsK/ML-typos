#!/usr/bin/env python
"""
build_typo_dataset.py  (LOCAL-ONLY VERSION)
-------------------------------------------

Create a typo-correction dataset **solely** from the corpora that already exist
under ./data:

1. SpellGramHugginFace.csv          (cols: source, target)
2. github-typo-corpus.v1.0.0.jsonl  (hugely compressed JSON-Lines)
3. kaggle/train.csv + kaggle/test.csv (cols: text, augmented_text)

The script:

• Harvests (original, corrected) pairs from those files.
• Adds a pool of *clean* sentences (perfectly correct) by taking the
  *target/text* side of each corpus.
• Generates synthetic errors over random clean sentences.
• Optionally derives token-level edit tags (KEEP / DELETE / REPLACE_x)
  using edit_tag_spellfix.tags.
• Writes CSV + JSON stats.

No network requests are made; everything is local.
"""

from __future__ import annotations
import argparse, csv, json, random, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Iterator, List, Tuple

# ---------- optional deps ----------
try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

try:
    from edit_tag_spellfix.tags import diff_to_tags, KEEP
except ImportError:
    diff_to_tags = None
    KEEP = "KEEP"

# ---------- CONST ----------
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR  = DATA_DIR / "cache"                 # still used for stats, etc.
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RE_ALPHA = re.compile(r"[A-Za-z]")

SPELL = SpellChecker() if SpellChecker else None


# ----------------------------------------------------------------------
# 0.  helpers
# ----------------------------------------------------------------------
def _is_eng(s: str) -> bool:
    return bool(RE_ALPHA.search(s))

def _valid_word(w: str) -> bool:
    if not _is_eng(w):
        return False
    if SPELL:
        return SPELL.correction(w.lower()) == w.lower()
    return True

# ----------------------------------------------------------------------
# 1.  load SpellGram CSV  (source → typo, target → correct)
# ----------------------------------------------------------------------
def iter_spellgram_pairs(limit: int | None = None) -> Iterator[Tuple[str, str]]:
    csv_path = DATA_DIR / "SpellGramHugginFace.csv"
    with csv_path.open(encoding="utf8") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            yield row["source"].strip(), row["target"].strip()
            if limit and (i + 1) >= limit:
                break

# ----------------------------------------------------------------------
# 2.  load Kaggle train/test  (augmented_text = typo, text = correct)
# ----------------------------------------------------------------------
def iter_kaggle_pairs(limit: int | None = None) -> Iterator[Tuple[str, str]]:
    kag_dir = DATA_DIR / "kaggle"
    for split in ("train.csv", "test.csv"):
        with (kag_dir / split).open(encoding="utf8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                yield row["augmented_text"].strip(), row["text"].strip()
                if limit:
                    limit -= 1
                    if limit == 0:
                        return

# ----------------------------------------------------------------------
# 3.  load GitHub JSONL  (each edit with is_typo==true → src = typo, tgt = correct)
# ----------------------------------------------------------------------
def iter_github_jsonl_pairs(limit: int | None = None) -> Iterator[Tuple[str, str]]:
    jl_path = (
        DATA_DIR
        / "github-typo-corpus.v1.0.0.jsonl"
        / "github-typo-corpus.v1.0.0.jsonl"
    )
    with jl_path.open(encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            for edit in obj["edits"]:
                if not edit.get("is_typo"):
                    continue
                src = edit["src"]["text"].strip()
                tgt = edit["tgt"]["text"].strip()
                if src and tgt and src != tgt:
                    yield src, tgt
                    if limit:
                        limit -= 1
                        if limit == 0:
                            return

# ----------------------------------------------------------------------
# 4.  synthetic typo injection
# ----------------------------------------------------------------------
KEYBOARD_ROWS = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./",
]
def _build_adj() -> dict[str, set[str]]:
    adj = defaultdict(set)
    for r, row in enumerate(KEYBOARD_ROWS):
        for c, ch in enumerate(row):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < len(KEYBOARD_ROWS) and 0 <= cc < len(KEYBOARD_ROWS[rr]):
                        n_ch = KEYBOARD_ROWS[rr][cc]
                        adj[ch.lower()].add(n_ch.lower())
                        adj[n_ch.lower()].add(ch.lower())
    return dict(adj)
ADJ = _build_adj()

def _inject_typo(token: str, rng: random.Random) -> str:
    if len(token) < 3 or not token.isalpha():
        return token
    strat = rng.choice(["swap", "sub", "delete", "insert"])
    chars = list(token)
    i = rng.randrange(len(chars))
    if strat == "swap" and i < len(chars) - 1:
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif strat == "sub":
        neigh = list(ADJ.get(chars[i].lower(), {chars[i]}))
        chars[i] = rng.choice(neigh)
    elif strat == "delete":
        del chars[i]
    elif strat == "insert":
        neigh = list(ADJ.get(chars[i].lower(), {chars[i]}))
        chars.insert(i, rng.choice(neigh))
    return "".join(chars)

def make_synthetic_pairs(clean_pool: List[str],
                         n_pairs: int,
                         rng: random.Random) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    while len(pairs) < n_pairs:
        sent = rng.choice(clean_pool)
        toks = sent.split()
        n_typos = max(1, int(0.15 * len(toks)))
        idxs = rng.sample(range(len(toks)), n_typos)
        altered = toks[:]
        for ix in idxs:
            altered[ix] = _inject_typo(altered[ix], rng)
        orig = " ".join(altered)
        if orig != sent:
            pairs.append((orig, sent))
    return pairs

# ----------------------------------------------------------------------
# 5.  tag pairs (optional)
# ----------------------------------------------------------------------
def _tag_pair(o: str, c: str) -> Tuple[str, str, str]:
    te = diff_to_tags(o.split(), c.split())
    return " ".join(te.tokens), c, " ".join(te.tags)

# ----------------------------------------------------------------------
# 6.  build dataset
# ----------------------------------------------------------------------
def build_dataset(args):
    rng = random.Random(args.seed)

    # ------------ A) REAL pairs ------------
    real_pairs = list(islice(
        iter_spellgram_pairs(), args.max_real // 3)) \
        + list(islice(iter_kaggle_pairs(), args.max_real // 3)) \
        + list(islice(iter_github_jsonl_pairs(), args.max_real // 3))

    rng.shuffle(real_pairs)
    real_pairs = real_pairs[: args.max_real]

    # ------------ B) CLEAN pool ------------
    clean_pool = [c for (_, c) in real_pairs]
    clean_pool += [row["text"].strip()                # from kaggle text
                   for row in csv.DictReader(
                       (DATA_DIR / "kaggle" / "train.csv").open(encoding="utf8"))]
    clean_pool = list({s for s in clean_pool if len(s.split()) > 3})
    rng.shuffle(clean_pool)

    clean_pairs  = [(s, s) for s in clean_pool[: args.max_clean]]
    synth_pairs  = make_synthetic_pairs(clean_pool, args.max_synth, rng)

    all_pairs = real_pairs + clean_pairs + synth_pairs
    rng.shuffle(all_pairs)

    # ------------ C) optional tagging ------------
    rows: List[Tuple[str, str, str | None]] = []
    if args.with_tags:
        if diff_to_tags is None:
            raise RuntimeError("edit_tag_spellfix.tags import failed; "
                               "--with_tags unavailable.")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_tag_pair, o, c): None for (o, c) in all_pairs}
            for fut in as_completed(futs):
                o, c, t = fut.result()
                rows.append((o, c, t))
    else:
        rows = [(o, c, None) for (o, c) in all_pairs]

    # ------------ D) write CSV ------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["original_text", "corrected_text"] +
                    (["edit_tags"] if args.with_tags else []))
        for o, c, t in rows:
            wr.writerow([o, c] if t is None else [o, c, t])

    stats = {
        "n_real":   len(real_pairs),
        "n_clean":  len(clean_pairs),
        "n_synth":  len(synth_pairs),
        "n_total":  len(rows),
    }
    json.dump(stats, out.with_suffix(".stats.json").open("w"), indent=2)
    print(f"✓ dataset saved to {out}  ({stats})")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="Path to output CSV, e.g. data/typo_dataset.csv")
    ap.add_argument("--max_real",  type=int, default=30_000)
    ap.add_argument("--max_clean", type=int, default=12_000)
    ap.add_argument("--max_synth", type=int, default=18_000)
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--with_tags", action="store_true")
    ap.add_argument("--seed",      type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    build_dataset(cli())
