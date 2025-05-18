"""
model_loader.py  –  now returns per-token tag + probability info

Public API used by app.py
-------------------------
• load_model() -> bool
• correct_text(sentence: str, top_k: int = 3)
      returns (corrected_sentence, elapsed_ms, corrections_made, token_info)
• symspell_model  (the loaded model object or None)
• MODEL_NAME
• SYMSPELL_DICTIONARY_PATH  (dummy for legacy log)
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from edit_tag_spellfix.model import RobertaTagger
from edit_tag_spellfix.tags  import KEEP, DELETE, is_replace, strip_prefix

# ------------------------------------------------------------------
# Paths / constants
# ------------------------------------------------------------------
MODEL_DIR  = Path(__file__).parent / "models" / "roberta_tag_60k_v2"
TAG_JSON   = MODEL_DIR / "tag2id.json"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "edit-tag-roberta-60k-v2"
SYMSPELL_DICTIONARY_PATH = "./dummy.txt"  # legacy

# Globals
symspell_model: RobertaTagger | None = None
_tokenizer     = None
_id2tag: Dict[int, str] | None = None

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _apply_tags(tokens: List[str], tags: List[str]) -> str:
    out = []
    for tok, tg in zip(tokens, tags):
        if tg == KEEP:
            out.append(tok)
        elif tg == DELETE:
            continue
        elif is_replace(tg):
            out.append(strip_prefix(tg))
        else:
            out.append(tok)
    return " ".join(out)


def _predict_verbose(sentence: str, top_k: int) -> tuple[str, List[dict]]:
    """Return corrected sentence and per-token info."""
    tokens = sentence.strip().split()
    batch  = _tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    word_ids = batch.word_ids(batch_index=0)
    enc = {k: v.to(DEVICE) for k, v in batch.items()}

    symspell_model.eval()
    with torch.no_grad():
        logits = symspell_model(**enc)["logits"]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [seq_len, num_tags]
    pred_ids = torch.argmax(probs, dim=-1).tolist()

    results = []
    tag_seq = []   # for sentence reconstruction
    for idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if idx == 0 or word_ids[idx-1] != w_id:   # first sub-token
            tag_id   = pred_ids[idx]
            tag_label = _id2tag[tag_id]
            tag_seq.append(tag_label)

            # top-k probs dict
            tk = min(max(1, top_k), probs.shape[-1])
            pk, pid = torch.topk(probs[idx], k=tk)
            dist = { _id2tag[int(i)]: float(p) for i, p in zip(pid, pk) }

            results.append({
                "token": tokens[w_id],
                "pred_tag": tag_label,
                "top_probs": dist
            })

    corrected = _apply_tags(tokens, tag_seq)
    return corrected, results

# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------
def load_model() -> bool:
    global symspell_model, _tokenizer, _id2tag
    if symspell_model is not None:
        return True
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, add_prefix_space=True)
        _id2tag    = {v: k for k, v in json.load(open(TAG_JSON)).items()}
        symspell_model = RobertaTagger.from_pretrained_with_tags(
            str(MODEL_DIR), TAG_JSON, freeze_encoder=True
        ).to(DEVICE)
        _ = _predict_verbose("warm up", top_k=1)  # quick warm-up
        return True
    except Exception as e:
        logging.getLogger(__name__).exception("Model load failed: %s", e)
        symspell_model = None
        return False


def correct_text(sentence: str, top_k: int = 3):
    """
    Returns:
        corrected_sentence (str)
        processing_time_ms (float)
        corrections_made   (bool)
        token_info         (list[dict]) – one entry per original token
    """
    if symspell_model is None:
        raise RuntimeError("Model not loaded")
    start = time.perf_counter()
    corrected, token_info = _predict_verbose(sentence, top_k)
    elapsed = (time.perf_counter() - start) * 1_000
    changed = corrected.strip() != sentence.strip()
    return corrected, elapsed, changed, token_info
