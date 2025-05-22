import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from edit_tag_spellfix.model import RobertaTagger
from edit_tag_spellfix.predict import predict_sentence


def main():
    ap = argparse.ArgumentParser(
        description="Create Data v3 via hard-negative mining: feed the model its own mistakes"
    )
    ap.add_argument("--model_dir", required=True, help="Trained edit-tag model directory")
    ap.add_argument("--csv", required=True, help="Existing dataset CSV")
    ap.add_argument("--out_csv", required=True, help="Output CSV with hard negatives added")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["original_text", "corrected_text"])

    tok = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)
    tag2id = json.load(open(Path(args.model_dir) / "tag2id.json"))
    id2tag = {v: k for k, v in tag2id.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RobertaTagger.from_pretrained_with_tags(
        args.model_dir, Path(args.model_dir) / "tag2id.json", freeze_encoder=True
    ).to(device)

    hard_negatives = []
    for orig, corr in tqdm(
        zip(df["original_text"], df["corrected_text"]),
        total=len(df),
        desc="Mining mistakes",
    ):
        pred = predict_sentence(orig, model, tok, id2tag, device)
        if pred.strip() != corr.strip():
            hard_negatives.append({"original_text": pred, "corrected_text": corr})

    if not hard_negatives:
        print("No mistakes found – dataset unchanged")
        df.to_csv(args.out_csv, index=False)
        return

    aug_df = pd.concat([df, pd.DataFrame(hard_negatives)], ignore_index=True)
    aug_df.to_csv(args.out_csv, index=False)
    print(f"Wrote Data v3 with {len(aug_df):,} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
