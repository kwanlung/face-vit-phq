# face_vit_phq/utils/early_stopping.py
from __future__ import annotations
from typing import Optional

class EarlyStopping:
    """
    Simple early-stopping utility.

    Args:
        patience:  epochs to wait after last improvement
        mode:      'min' (e.g. val_loss)  or  'max' (e.g. val_accuracy)
        delta:     minimal change considered an improvement
    """
    def __init__(self, patience: int = 5, mode: str = "min", delta: float = 0.0):
        assert mode in ("min", "max")
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best: Optional[float] = None
        self.count = 0
        self.stop = False
        self.is_best = False

    def __call__(self, metric: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best is None:                       # first call
            self.best = metric
            self.is_best = True
            return False

        # has the metric improved?
        improvement = (metric < self.best - self.delta) if self.mode == "min" else (metric > self.best + self.delta)


        if improvement:
            self.best = metric
            self.count = 0
            self.is_best = True
        else:
            self.count += 1
            self.is_best = False
            if self.count >= self.patience:
                self.stop = True
        return self.stop
