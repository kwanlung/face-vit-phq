#!/usr/bin/env python
# face_vit_phq/eval.py
"""
Evaluate a saved checkpoint on the test split.

Usage:
    python -m face_vit_phq.eval --ckpt outputs/best_model.pth --stage finetune
"""
from __future__ import annotations
import argparse, torch, numpy as np
from pathlib import Path

from .configs import cfg
from data import dataset
from models import vit_model
from utils.metrics import mse, mae, concordance_corr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to .pth checkpoint")
    p.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.stage == "pretrain":
        model = vit_model.EmotionViTClassifier(num_classes=cfg.model.num_classes,
                                               model_name=cfg.model.name,
                                               pretrained=False)
    else:
        model = vit_model.PHQViTRegressor(model_name=cfg.model.name,
                                          pretrained_backbone=False)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # DataLoader (test split)
    if args.stage == "pretrain":
        _, test_loader = dataset.get_combined_emotion_dataset(
            cfg.training.batch_size, cfg.data.image_size)
    else:
        test_loader = dataset.get_phq_dataloaders(
            cfg.training.batch_size, cfg.data.image_size)[1]   # idx=1 is val; swap if you have test

    # Evaluation
    all_out, all_tgt = [], []
    for imgs, tgt in test_loader:
        out = model(imgs.to(device))
        all_out.append(out.cpu())
        all_tgt.append(tgt.cpu())
    preds = torch.cat(all_out)
    gts   = torch.cat(all_tgt)

    if args.stage == "pretrain":
        acc = (preds.argmax(1) == gts).float().mean().item()
        print(f"Test accuracy: {acc:.4f}")
    else:
        mse_val = mse(preds, gts)
        mae_val = mae(preds, gts)
        ccc_val = concordance_corr(preds, gts)
        print(f"Test  MSE={mse_val:.3f} | MAE={mae_val:.3f} | CCC={ccc_val:.3f}")

if __name__ == "__main__":
    main()
# This script is used to evaluate a saved model checkpoint on the test dataset.
# It loads the model, prepares the test data, and computes evaluation metrics.