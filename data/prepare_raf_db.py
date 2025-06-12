#!/usr/bin/env python
"""
prepare_raf_db.py
────────────────────────────────────────────────────────────────────────────
Clean RAF-DB-TCC so it matches the 7-emotion setup used for FER+ / AffectNet.

* Splits the original 5-element “label” list into:
    emotion : ClassLabel(7)
    bbox    : Sequence[float32] length 4
* Maps RAF codes → ['anger','disgust','fear','happiness','neutral','sadness','surprise']
* Saves locally (optional) and pushes to HF Hub (default: deanngkl/raf-db-7emotions).

Run:
  python -m face_vit_phq.data.prepare_raf_db          # train split
  python -m face_vit_phq.data.prepare_raf_db --split val
  python -m face_vit_phq.data.prepare_raf_db --push-to deanngkl/raf-db-7emotions
"""

from __future__ import annotations
import argparse, os, pathlib, sys
from typing import Dict, List

from datasets import (
    load_dataset, Dataset, Features, ClassLabel,
    Sequence, Value
)

TARGET_EMOTIONS: List[str] = [
    "anger", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
]

# RAF original → canonical index
RAF_TO_CANON = {
    6: 0,  # id6 angry     → anger
    3: 1,  # id3 disgust   → disgust
    2: 2,  # id2 fear      → fear
    4: 3,  # id4 happiness → happiness
    # neutral (index 4 in the canonical list) is **missing** in this dataset
    5: 5,  # id5 sadness   → sadness
    1: 6,  # id1 surprise  → surprise
}

def split_and_remap(row):
    """
    RAF-DB row["label"] is a 5-tuple: [cid, x, y, w, h]
    We:
      • pop the original list,
      • map cid → canonical,
      • keep bbox untouched,
      • write the new scalar label back.
    """
    raw = row.pop("label")            # ⚠️ remove BEFORE we overwrite
    cid  = int(raw[0])                # original class id
    if cid not in RAF_TO_CANON and (cid - 1) in RAF_TO_CANON:
        cid -= 1                      # rare 1-based variant safeguard

    row["bbox"]  = [float(x) for x in raw[1:]]           # 4 floats
    row["label"] = RAF_TO_CANON[cid]                     # scalar int
    return row


def build_new_features(old_feats) -> Features:
    """Return a Features object with emotion(ClassLabel) + bbox."""
    feats = dict(old_feats)  # shallow copy of the original dict
    feats.pop("label", None)  # we removed it
    feats["label"] = ClassLabel(names=TARGET_EMOTIONS)
    feats["bbox"]    = Sequence(feature=Value("float32"), length=4)
    return Features(feats)

# ----------------------------------------------------------------------
def convert(split: str, out_dir: pathlib.Path | None) -> Dataset:
    print(f"▶ Loading Mat303/raf-db-tcc::{split}")
    ds = load_dataset("Mat303/raf-db-tcc", split=split, streaming=False)

    print("▶ Splitting columns & remapping labels")
    ds = ds.map(split_and_remap,
                desc="Process RAF rows",
                num_proc=os.cpu_count() or 1)

    ds = ds.cast(build_new_features(ds.features))  # recast with ClassLabel

    # sanity check
    eid = ds[0]["label"]
    print(f"✓ Sample after cast → int:{eid}  str:{ds.features['label'].int2str(eid)}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(out_dir.as_posix())
        print(f"💾 Saved to {out_dir.resolve()}")

    return ds

def push(ds: Dataset, repo_id: str, token: str):
    print(f"🚀 Pushing to {repo_id}")
    ds.push_to_hub(repo_id, token=token)
    print("✓ Push complete")

# ----------------------------------------------------------------------
def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--out-dir", default="local_data/raf_db_clean",
                        help="Local save path (omit or --no-save to skip)")
    parser.add_argument("--no-save", action="store_true", help="Skip local save")
    parser.add_argument("--push-to", default="deanngkl/raf-db-7emotions",
                        help="HF repo to push (empty string ⇒ skip)")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"),
                        help="Write token for HF Hub")
    args = parser.parse_args()

    out = None if args.no_save else pathlib.Path(args.out_dir)
    ds  = convert(args.split, out)

    push(ds, args.push_to, args.hf_token)

if __name__ == "__main__":
    cli()
