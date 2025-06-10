# face_vit_phq/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
from omegaconf import OmegaConf, MISSING

# ───────── Section dataclasses ─────────
@dataclass
class AugmentationCfg:
    horizontal_flip: bool = True
    random_rotation: bool = True
    rotation: int = 10
    color_jitter: List[float] = field(
        default_factory=lambda: [0.2, 0.2, 0.2, 0.1])
    random_crop: float = 0.9

@dataclass
class EmotionSourceCfg:
    source: str = "hf" #source: Literal["hf", "local"] = "hf"
    repo: Optional[str] = None
    split: str = "train"
    path: Optional[str] = None
    label_csv: Optional[str] = None

@dataclass
class PhqSourceCfg:
    source: str = "local" #source: Literal["hf", "local"] = "local"
    path: Optional[str] = None
    repo: Optional[str] = None
    split: Optional[str] = None
    label_csv: str = MISSING
    splits: Dict[str, str] = field(default_factory=dict)

@dataclass
class DataCfg:
    emotion_sources: List[EmotionSourceCfg] = field(default_factory=list)
    emotion_val_split: float = 0.1
    cache_dir: str = "./cache"
    image_size: int = 224
    augmentations: AugmentationCfg = field(default_factory=AugmentationCfg)
    phq_source: PhqSourceCfg = field(default_factory=PhqSourceCfg)
    affectnet_no_contempt: str = "face_vit_phq/data/affectnet_no_contempt"  # Path to the preprocessed AffectNet dataset without contempt
    label_mapping: Dict[str, str] = field(default_factory=lambda: {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happiness": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "surprise": "surprise",
    })

@dataclass
class RegressionCfg:
    use_ccc_loss: bool = False
    freeze_backbone: bool = False

@dataclass
class ModelCfg:
    name: str = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 7
    drop_rate: float = 0.1
    regression: RegressionCfg = field(default_factory=RegressionCfg)
    global_pool: str = "token" #global_pool: Literal["token", "avg"] = "token"

@dataclass
class TrainingCfg:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine" #scheduler: Literal["cosine", "step"] = "cosine"
    warmup_epochs: int = 1
    mixed_precision: bool = True
    early_stopping_patience: int = 3
    log_interval: int = 50
    init_backbone_from: str = "./checkpoints/emotion_vit_best.pt"
    output_dir: str = "./outputs"
    use_profiler: bool = False
    balanced_sampler: bool = True
    cutmix_prob: float = 0.5
    grad_clip: float = 1.0
    tf32: bool = True  # Enable NVIDIA TensorFloat32 (TF32) acceleration

@dataclass
class LoggingCfg:
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "FacePHQ"
    logging_dir: str = "face_vit_phq/logs"

@dataclass
class RootCfg:
    stage: str = "pretrain" #stage: Literal["pretrain", "finetune"] = "pretrain"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

# ───────── Helper loader ─────────
def load_cfg(yaml_path: str = "face_vit_phq/configs/config.yaml") -> RootCfg:
    base = OmegaConf.structured(RootCfg)          # typed defaults
    yaml = OmegaConf.load(yaml_path)
    merged = OmegaConf.merge(base, yaml)          # YAML overrides
    return OmegaConf.to_object(merged)            # dataclass tree

# global singleton
cfg: RootCfg = load_cfg()
