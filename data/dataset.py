import os, csv
from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from datasets import load_dataset, load_from_disk

#----------global config----------#
from ..configs import cfg   # cfg is already loaded dataclass

# ----------transforms----------#
def get_transforms(split="train", image_size=224):
    aug = cfg.data.augmentations
    if split == "train":
        t = []
        #note: torchvision Random* transforms return self, so we append conditionally
        t.append(transforms.Resize((image_size, image_size)))
        if getattr(aug, "horizontal_flip", False):
            t.append(transforms.RandomHorizontalFlip())
        if getattr(aug, "random_rotation", False):
            t.append(transforms.RandomRotation(aug.rotation))
        if getattr(aug, "color_jitter", False):
            t.append(transforms.ColorJitter(*aug.color_jitter))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))  # normalize to [-1,1]
        return transforms.Compose(t)
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
#-----------HF wrapper-----------#
class HFImageDataset(Dataset):
    "Wraps a Hugging Face dataset that returns {image:<PTL>, label:<int>}"
    def __init__(self, 
                 repo, 
                 split, 
                 transform, 
                 cache_dir, 
                 label_key="label", 
                 image_key="image", 
                 label_mapping=None):
        super().__init__()
        # Use load_from_disk if repo is a local directory
        print(f"Loading dataset from {repo} (split={split}) ...")
        self.ds = load_dataset(repo, split=split, cache_dir=cache_dir)
        print(f"Loaded dataset from {repo} (split={split}), length={len(self.ds)}")
        # if "AffectNet" in repo:
        #     self.ds = self.ds.filter(lambda x: x[label_key] != 7)
        self.transform = transform
        self.label_key = label_key
        self.image_key = image_key
        self.label_mapping = label_mapping
        
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, idx):
        rec = self.ds[idx]
        img = rec[self.image_key].convert("RGB")
        label = rec[self.label_key]
        # Handle label as list or scalar
        if isinstance(label, list):
            if all(isinstance(x, (int, float)) for x in label):
                label = int(label.index(max(label)))
            else:
                label = int(label[0])
        else:
            label = int(label)
            
        # Apply label mapping
        if self.label_mapping:
            label = self.label_mapping.get(label, label)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    
#-----------Local CSV wrapper-----------#
class CSVDataset(Dataset):
    "Dataset where a CSV maps <filename, <numeric_label>"
    def __init__(self, root_dir, csv_file, transform, label_type="init", label_mapping=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = label_mapping
        self.samples = []
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            for fn, lab in reader:
                lab = float(lab) if label_type == "float" else int(lab)
                # Apply label mapping if provided
                if self.label_mapping:
                    lab = self.label_mapping.get(lab, lab)
                self.samples.append((fn,lab))
                
    def __len__(self):return len(self.samples)
    
    def __getitem__(self, idx):
        fn, lab = self.samples[idx]
        img = Image.open(os.path.join(self.root_dir, fn)).convert("RGB")
        if self.transform: img = self.transform(img)
        dtype = torch.float32 if isinstance(lab, float) else torch.int64
        return img, torch.tensor(lab, dtype=dtype)
    
def build_emotion_dataset(cfg_entry, split, transform, label_mapping=None):
    if cfg_entry.source == "hf":
        return HFImageDataset(cfg_entry.repo, 
                              split, 
                              transform, 
                              cfg.data.cache_dir, 
                              label_mapping=label_mapping)
    elif cfg_entry.source == "local":
        return CSVDataset(cfg_entry.path, 
                          cfg_entry.label_csv, 
                          transform, 
                          label_type="int",
                          label_mapping=label_mapping)
    else:
        raise ValueError(f"Unknown source {cfg_entry.source}")
    
def build_phq_dataset(split, transform):
    phq = cfg.data.phq_source
    if phq.source == "local":
        list_file = phq.splits[split]
        # build a split‑specific CSV on‑the‑fly:
        with open(list_file) as f: files = [ln.strip() for ln in f]
        # create a temp list of (file, score)
        mapping = {}
        with open(phq.label_csv) as f:
            for fn, score in csv.reader(f):
                mapping[fn] = float(score)
        temp_csv = [ (fn, mapping[fn]) for fn in files if fn in mapping ]
        # wrap in in‑memory dataset:
        class InMem(Dataset):
            def __len__(self): return len(temp_csv)
            def __getitem__(self, idx):
                fn, score = temp_csv[idx]
                img = Image.open(os.path.join(phq.path, fn)).convert("RGB")
                if transform: img = transform(img)
                return img, torch.tensor(score, dtype=torch.float32)
        return InMem()
    elif phq.source == "hf":
        return HFImageDataset(phq.repo, phq.split, transform, cfg.data.cache_dir,
                              label_key="phq_score")
    else:
        raise ValueError(f"Unknown phq source {phq.source}")


# -------- helper to create WeightedRandomSampler ---------

def _make_balanced_sampler(dataset: Dataset, num_classes: int):
    labels = []
    for _, lab in dataset:
        labels.append(int(lab))
    labels = torch.tensor(labels)
    class_count = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / class_count[labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

#-----------Public API (used by train.py)-----------#
def get_combined_emotion_dataset(batch_size, image_size):
    "Returns (train_loader, val_loader) concatenating all emotion datasets"
    label_mapping = cfg.data.label_mapping
    train_sets, val_sets = [],[]
    for src in cfg.data.emotion_sources:
        # Load the dataset once
        full_ds = build_emotion_dataset(src, 
                                             src.split, 
                                             get_transforms("train", image_size),
                                             label_mapping)
        print(f"Loaded {len(full_ds)} samples from {src.repo}")
        
        # Try to load validation split, fallback to manual split if not available
        try:
            # Try to load validation split
            val_ds = build_emotion_dataset(src, "validation", get_transforms("val", image_size), label_mapping)
            print(f"Loaded {len(val_ds)} validation samples from {src.repo}")
        except Exception as e:
            # If validation split not available, do manual split
            print(f"No explicit validation split for {src.repo}, doing manual split.")
            val_size = int(len(full_ds) * cfg.data.emotion_val_split)
            full_ds, val_ds = random_split(full_ds, 
                                           [len(full_ds) - val_size, val_size])
            train_sets.append(full_ds)
            val_sets.append(val_ds)
    train_concat = ConcatDataset(train_sets)
    val_concat = ConcatDataset(val_sets)
    # Concatenate all datasets
    if cfg.training.balanced_sampler:
        sampler = _make_balanced_sampler(train_concat, cfg.model.num_classes)
        train_loader = DataLoader(train_concat,
                                  batch_size= batch_size,
                                  sampler=sampler,
                                  num_workers=4)
    else:
        train_loader = DataLoader(train_concat,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)
    val_loader = DataLoader(val_concat,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
    print(f"Loaded {len(train_loader.dataset)} training samples from {len(train_sets)} sources")
    print(f"Loaded {len(val_loader.dataset)} validation samples from {len(val_sets)} sources")
    # Check if any dataset is empty
    if len(train_loader.dataset) == 0:
        raise ValueError("No training samples found in any dataset.")
    if len(val_loader.dataset) == 0:
        raise ValueError("No validation samples found in any dataset.")
    return train_loader, val_loader

def get_phq_dataloaders(batch_size, image_size):
    train_ds = build_phq_dataset("train", get_transforms("train", image_size))
    val_ds   = build_phq_dataset("val",   get_transforms("val",   image_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

# ───────── CLI smoke-test ─────────
if __name__ == "__main__":
    tl, vl = get_combined_emotion_dataset(cfg.training.batch_size, cfg.data.image_size)
    print("Emotion batch", next(iter(tl))[0].shape)
    # tl2, vl2 = get_phq_dataloaders(8, cfg.data.image_size)
    # print("PHQ batch", next(iter(tl2))[0].shape)