# prepare_affectnet.py  -- trimmed but still yours
from datasets import load_dataset, Features, ClassLabel
from pathlib import Path
import datetime, argparse

NEW_NAMES = ['anger','disgust','fear','happiness','neutral','sadness','surprise']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("local_data/affectnet_no_contempt"))
    ap.add_argument("--hub-repo", default=None,
                    help="deanngkl/affectnet_no_contempt")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    t0 = datetime.datetime.now()
    print(f"[{t0:%F %T}] start")

    ds = load_dataset("chitradrishti/AffectNet", split="train")
    ds = ds.filter(lambda x: x["label"] != 1, num_proc=None)
    ds = ds.map(lambda x: {"label": x["label"] - 1 if x["label"] > 1 else x["label"]},
                num_proc=None)

    feats = ds.features
    feats["label"] = ClassLabel(names=NEW_NAMES)
    ds = ds.cast(feats)

    ds.save_to_disk(args.out_dir)
    print(f"saved → {args.out_dir}  ({len(ds):,} rows, fp={ds._fingerprint})")

    if args.push:
        from huggingface_hub import HfApi
        HfApi().create_repo(repo_id=args.hub_repo,
                            repo_type="dataset",
                            private=args.private,
                            exist_ok=True)
        ds.push_to_hub(args.hub_repo, split="train")
        print("pushed ✅")

    print(f"[{datetime.datetime.now():%F %T}] done")

if __name__ == "__main__":
    main()
