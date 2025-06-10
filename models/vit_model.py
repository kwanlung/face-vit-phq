"""A single-file implementation of three closely related Vision-Transformer models:

* **ViTBackbone** - generic feature extractor (CLS-token or avg-pooled patches).
* **EmotionViTClassifier** - 7-way emotion recognition head.
* **PHQViTRegressor** - scalar regression head for PHQ-score prediction.

Design goals
------
* **Modular** - backbone is swappable; heads are tiny, independent modules.
* **Fine-tune friendly** - one-call helpers to freeze/unfreeze the ViT.
* **Config driven** - factory helpers make integration with OmegaConf configs trivial.
* **Transfer learning ready** - regression head can optionally warm-start from
  a checkpoint of the emotion-finetuned model.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union, Any
import pathlib

import torch
import torch.nn as nn
import timm # Torchvision also has ViT, but timm provides convenient model builder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────── backbone ─────────────────────────────────────────
class ViTBackbone(nn.Module):
    """Vision-Transformer backbone that directly outputs pooled features `(B, D)`.

    Parameters
    ----------
    model_name : str, default "vit_base_patch16_224"
        Any ViT family model recognised by **timm**.
    pretrained : bool, default True
        Load ImageNet-1K weights.
    global_pool : {"token", "avg"}
        • **token** - return the `[CLS]` token embedding.  
        • **avg**   - mean-pool the patch tokens; sometimes boosts robustness on
          tightly-cropped faces.
    """
    def __init__(self, 
                 model_name: str = "vit_base_patch16_224",
                 *,
                 pretrained: bool = True,
                 global_pool: str = "token",
                 ) -> None:
        super().__init__()
        if global_pool not in {"token", "avg"}:
            raise ValueError(f"global_pool must be 'token' or 'avg', got {global_pool}")
        # Load a ViT model. By default, timm returns a model with a classification head.
        self.vit = timm.create_model(model_name, 
                                     pretrained=pretrained, # We will load pretrained weights later
                                     num_classes=0, 
                                     global_pool=global_pool)
        # The timm ViT model has an attribute called 'head' for the classifier.
        # We will replace or remove it in task-specific subclasses.
        self.out_features: int = self.vit.num_features # e.g. 768 for ViT‑Base. This is the size of the output features (CLS token)
        
    # fine‑tune helpers
    @torch.no_grad()
    def freeze(self) -> None:
        "Disable gradient updates for every parameter in the backbone."
        for param in self.parameters():
            param.requires_grad = False
        self.eval() # Set the model to evaluation mode
        
    @torch.no_grad()
    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x) # (B, 3, H, W) ➞ (B, D) where D is the output size of the ViT backbone

# ───────────────────────────────────── emotion head ────────────────────────────────────────
class EmotionViTClassifier(nn.Module):
    def __init__(self,
                 *,
                 num_classes: int = 7,
                 drop_rate: float = 0.1,
                 freeze_backbone: bool = False,
                 model_name: str = "vit_base_patch16_224",
                 pretrained: bool = True,
                 global_pool: str = "token", 
                 ) -> None:
        super().__init__()
        
        # Backbone (feature extractor)
        self.backbone = ViTBackbone(model_name=model_name, 
                                    pretrained=pretrained, 
                                    global_pool=global_pool)
        if freeze_backbone:
            self.backbone.freeze()
            
        # Task head: LN → Dropout → Linear
        self.norm = nn.LayerNorm(self.backbone.out_features)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        # The classifier is a linear layer that maps the features to the number of classes.
        self.classifier = nn.Linear(self.backbone.out_features, num_classes) # num_features is the output size of the ViT backbone
        self._init_head() # Initialise the classifier weights

    # Helpers
    def _init_head(self) -> None:
        "HeKaiMing initialisation ensures a healthy start for the classifier."
        # Reset the classifier weights using He initialization
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        nn.init.zeros_(self.classifier.bias)

    # API
    def forward_features(self, x:torch.Tensor) -> torch.Tensor: # (B, 3, H, W) → (B, D)
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, 3, H, W) → (B, 7)
        # Get features from the ViT backbone
        features = self.dropout(self.norm(self.forward_features(x).float()))
        # Pass features through the classifier
        return self.classifier(features)
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)

# ───────────────────────────────────── regression head ─────────────────────────────────────
class PHQViTRegressor(nn.Module):
    """ViT encoder + lightweight regression head for PHQ‑score prediction.

    Parameters
    ----------
    out_dim : int, default 1
        Number of regression targets.
    drop_rate : float, default 0.1
        Dropout applied after the backbone.
    freeze_backbone : bool
        If ``True``, backbone parameters are frozen (useful for small datasets).
    init_backbone_from : str or ``pathlib.Path`` or ``None``
        Checkpoint path whose *backbone* weights will initialise this model.
        Must be a state-dict produced by :class:`EmotionViTClassifier` or a
        compatible ViT.  Loaded with ``strict=False`` so the new regressor head
        stays randomly initialised.
    **backbone_kwargs : Any
        Forwarded to :class:`ViTBackbone` (e.g. ``model_name``, ``pretrained``).
    """
    def __init__(self,
                 *,
                 out_dim: int = 1,
                 drop_rate: float = 0.1,
                 freeze_backbone: bool = False,
                 init_backbone_from: Optional[Union[str, pathlib.Path]] = None,
                 model_name: str = "vit_base_patch16_224",
                 pretrained: bool = True,
                 global_pool: str = "token",
                 ) -> None:
        super().__init__()
        
        # Backbone
        self.backbone = ViTBackbone(model_name=model_name, 
                                    pretrained=pretrained, 
                                    global_pool=global_pool)
        if freeze_backbone:
            self.backbone.freeze()
            
        # optionally warm-start backbone from a checkpoint
        if init_backbone_from is not None:
            # ckpt_path =pathlib.Path(init_backbone_from)
            # state_dict = torch.load(ckpt_path, map_location='cpu')
            # missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            # if unexpected:
            #     print(f"[PHQViTRegressor] Ignored keys: {unexpected[:5]} …")
            # if missing:
            #     print(f"[PHQViTRegressor] Missing keys: {missing[:5]} …")
            ckpt = torch.load(pathlib.Path(init_backbone_from), map_location="cpu")
            self.backbone.load_state_dict(ckpt, strict=False)  # ignore head keys
                
        # Head: LN -> Dropout > Linear
        self.norm = nn.LayerNorm(self.backbone.out_features)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.regressor = nn.Linear(self.backbone.out_features, out_dim)
        self._init_head()
        
    def _init_head(self) -> None:
        nn.init.normal_(self.regressor.weight, std=0.02)
        nn.init.zeros_(self.regressor.bias)
        
    # Front API
    def forward_features(self, x:torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) ➞ (B, out_dim)
        feats = self.dropout(self.norm(self.forward_features(x).float()))
        return self.regressor(feats).squeeze(-1)  # (B,) if out_dim==1

def build_classifier_from_cfg(cfg: Any)->EmotionViTClassifier:
    return EmotionViTClassifier(
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.drop_rate,
        freeze_backbone=cfg.model.regression.freeze_backbone,
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        global_pool=cfg.model.global_pool,
    )
    
def build_regressor_from_cfg(cfg: Any)->PHQViTRegressor:
    return PHQViTRegressor(
        out_dim=1,
        drop_rate=cfg.model.drop_rate,
        freeze_backbone=cfg.model.regression.freeze_backbone,
        init_backbone_from=cfg.training.init_backbone_from,
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        global_pool=cfg.model.global_pool,
    )

__all__ = [
    "ViTBackbone",
    "EmotionViTClassifier",
    "PHQViTRegressor",
    "build_classifier_from_cfg",
    "build_regressor_from_cfg",
]

# model_emotion = EmotionViTClassifier()
# logger.info(f"Model Emotion: {model_emotion}")
# model_phq = PHQViTRegressor()
# logger.info(f"Model PHQ: {model_phq}")