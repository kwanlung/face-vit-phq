# face_vit_phq/utils/logger.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

class BaseLogger:
    def log_scalar(self, tag: str, value: float, step: int): ...
    def log_image(self, tag: str, img, step: int): ...
    def close(self): ...

# ───────────────── TensorBoard ─────────────────
class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, img, step):
        # img: np.ndarray HxWxC or CHW (0–255 or 0–1)
        self.writer.add_image(tag, img, step, dataformats='HWC')

    def close(self):
        self.writer.close()

# ───────────────── Weights & Biases ─────────────────
class WandBLogger(BaseLogger):
    def __init__(self, project: str, run_name: Optional[str] = None):
        import wandb
        self.wandb = wandb
        wandb.init(project=project, name=run_name)

    def log_scalar(self, tag, value, step):
        self.wandb.log({tag: value}, step=step)

    def log_image(self, tag, img, step):
        self.wandb.log({tag: self.wandb.Image(img)}, step=step)

    def close(self):
        self.wandb.finish()
