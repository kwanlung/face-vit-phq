from __future__ import annotations
from typing import Tuple, List, Dict, Any
import os, time, datetime
import numpy as np

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.cuda.amp import autocast, GradScaler
from data import dataset
from models import vit_model
from utils.losses import CCCLoss
from utils.metrics import concordance_corr, mae, mse
from utils.logger import TensorBoardLogger, WandBLogger
from utils.early_stopping import EarlyStopping
from configs import cfg
from sklearn.metrics import classification_report  # Import after validation loop

# -------------------- helper for CutMix --------------------

def rand_bbox(size: Tuple[int,int,int,int], lam: float):
    """Random square bbox; size = (B, C, H, W)."""
    _, _, H, W = size
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for faster training on Ampere GPUs
    
    if cfg.stage == "pretrain":
        # Prepare data loaders
        train_loader, val_loader = dataset.get_combined_emotion_dataset(
            cfg.training.batch_size,
            cfg.data.image_size,
            # use_weighted_sampler=True  # Add this flag to enable WeightedRandomSampler
        )
        model = vit_model.build_classifier_from_cfg(cfg)
        criterion = torch.nn.CrossEntropyLoss()
    # else:
    #     # Fine-tunign stage
    #     # Use the same transforms as val (minimal augmentation) for training in regression to avoid altering facial cues too much, though one could still augment slightly.
    #     train_transform = dataset.get_transforms("train", cfg.data.image_size)
    #     val_transform = dataset.get_transforms("val", cfg.data.image_size)
    #     train_ds = dataset.get_phq_dataloaders(cfg.data.phq_faces_dir, split="train", transform=train_transform)
    #     val_ds = dataset.get_phq_dataloaders(cfg.data.phq_faces_dir, split="val", transform=val_transform)
    #     train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    #     val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)
    #     model = vit_model.build_regressor_from_cfg(model_name=cfg.model.name, pretrained=True, drop_rate=cfg.model.drop_rate)
    #     # Loss function
    #     criterion = CCCLoss() if cfg.model.regression.use_ccc_loss else torch.nn.MSELoss()
    
    # Move model to device
    model = model.to(device)
    
    epochs = cfg.training.epochs
    warmup_epochs = cfg.training.warmup_epochs
    base_lr = cfg.training.learning_rate
    
    optimiser_cls = AdamW if cfg.training.weight_decay else Adam
    
    # Optimizer and scheduler
    optimizer = optimiser_cls(model.parameters(), 
                     lr=base_lr, 
                     weight_decay=cfg.training.weight_decay)
    
    # Set up learning rate schedular
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=(epochs - warmup_epochs) * len(train_loader),
        eta_min=1e-6,
    )
    
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3, # start from 0 during warmup
            end_factor=1,   # reach base_lr
            total_iters=warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = cosine
    print(f"Using optimizer: {optimizer.__class__.__name__} with learning rate {base_lr}")
        
    # Mixed precision scaler
    scaler = GradScaler(enabled=cfg.training.mixed_precision)
    
    # Logging setup
    logger = None
    if cfg.logging.use_tensorboard:
        logger = TensorBoardLogger(log_dir=os.path.join(cfg.logging.logging_dir, "logs"))
    if cfg.logging.use_wandb:
        logger = WandBLogger(project=cfg.logging.wandb_project)
        
    # Early stopping setup
    early_stopping = EarlyStopping(patience=cfg.training.early_stopping_patience,
                                   mode=('min' if cfg.stage=="finetune" else 'max'),)
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    print("DataLoader sanity-check …")
    # Test DataLoader speed with a single batch
    images, targets = next(iter(train_loader))
    print(f"First batch loaded: {images.shape}, targets: {targets.shape}, Unique targets: {targets.unique()}")
    print("DataLoader test passed. Starting training loop...")
    for epoch in range(1, epochs + 1):
        epoch_start_time = datetime.datetime.now()
        print(f"Epoch {epoch} started at {epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            
            # —— optional CutMix ——
            if cfg.training.cutmix_prob and np.random.rand() < cfg.training.cutmix_prob:
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(images.size(0)).to(device)
                bbx1,bby1,bbx2,bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                target_a, target_b = targets, targets[rand_index]
            else:
                lam = None
            
            # forward pass with mixed precision if enabled
            with autocast(enabled=cfg.training.mixed_precision):
                outputs = model(images)
                if lam is not None:
                    loss = lam * criterion(outputs, target_a) + (1-lam) * criterion(outputs, target_b)
                else:
                    loss = criterion(outputs, targets)
            # backward pass
            scaler.scale(loss).backward()
                    
            # grad‑clip
            if cfg.training.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)        
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # if not using scaler: loss.backward() & optimizer.step()
            train_loss += loss.item() * images.size(0)
            if cfg.stage == "pretrain":
                preds = outputs.argmax(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
            # Logging every few batches
            if logger and batch_idx % cfg.training.log_interval ==0:
                avg_loss = train_loss / ((batch_idx + 1) * images.size(0))
                if cfg.stage == "pretrain":
                    logger.log_scalar("train/loss", avg_loss, epoch * len(train_loader) + batch_idx)
                else:
                    logger.log_scalar("train/loss", avg_loss, epoch * len(train_loader) + batch_idx)
                # (We can also log running accuracy here for classification)

        # Compute average training loss and accuracy (if classification) for the epoch
        avg_train_loss = train_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start
        epoch_end_time = datetime.datetime.now()
        hours, rem = divmod(epoch_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        print(f"Epoch {epoch} finished at {epoch_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if cfg.stage == "pretrain":
            train_acc = correct / total if total else 0.0
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Time: {time_str}")
        else:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, Time: {time_str}")
        print(f"LR end-of-epoch: {scheduler.get_last_lr()[0]:.2e}")  # Log LR at the end of each epoch
            
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds: List[int] = []
        all_targs: List[int] = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                # gather all outputs and targets for metrics
                if cfg.stage == "pretrain":
                    preds = outputs.argmax(1)
                    val_correct += (preds == targets).sum().item()
                    val_total   += targets.size(0)
                    all_preds.extend(preds.cpu().tolist())
                    all_targs.extend(targets.cpu().tolist())
                else:
                    all_preds.extend(outputs.cpu().numpy().tolist())
                    all_targs.extend(targets.cpu().numpy().tolist())
        val_loss /= len(val_loader.dataset)
        if cfg.stage == "pretrain":
            val_acc = val_correct / val_total
            metric   = val_acc
        else:
            # for regression you might swap in CCC here
            val_acc = -val_loss  # higher better convention for stopper
            metric  = val_acc
            
        # ---------- logging ----------
        print(f"Epoch [{epoch}/{epochs}]  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  LR={lr_now:.2e}")
        if cfg.stage == "pretrain" and epoch % 5 == 0:
            print(classification_report(all_targs, all_preds, digits=3))
        if logger:
            logger.log_scalar("train/loss", avg_train_loss, epoch)
            logger.log_scalar("val/loss", val_loss, epoch)
            if cfg.stage == "pretrain":
                logger.log_scalar("train/acc", train_acc, epoch)
                logger.log_scalar("val/acc", val_acc, epoch)
            logger.log_scalar("LR", lr_now, epoch)
            
        
        # Save the best model if improved
        if early_stopping.is_best:
            best_path = os.path.join(cfg.training.output_dir, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved at {best_path}")
        # Early stopping check
        if early_stopping(metric):
            print(f"Early stopping at epoch {epoch}")
            break
    # save final
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    final_path = os.path.join(cfg.training.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved at {final_path}")
    if logger:
        logger.close()
    print("Training complete.")
    
if __name__ == "__main__":
    train()