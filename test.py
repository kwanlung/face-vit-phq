import torch
from datasets import load_dataset
from collections import Counter

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA not available")

affectnet_ds = load_dataset("deanngkl/affectnet_no_contempt", split="train")
label_names = affectnet_ds.features["label"].names
print(f"deanngkl/affectnet_no_contempt's Label mapping: {label_names}")
affectnet_label_counts = Counter(affectnet_ds["label"])
print(f"Label distribution of deanngkl/affectnet_no_contempt: {affectnet_label_counts}")
print(" ")
fer_ds = load_dataset("deanngkl/ferplus-7cls", split="train")
label_names = fer_ds.features["label"].names
print(f"deanngkl/ferplus-7cls's Label mapping: {label_names}")
fer_label_counts = Counter(fer_ds["label"])
print(f"Label distribution of deanngkl/ferplus-7cls: {fer_label_counts}")
print(" ")
raf_ds = load_dataset("deanngkl/raf-db-7emotions", split="train")
label_names = raf_ds.features["label"].names
print(f"deanngkl/raf-db-7emotions's Label mapping: {label_names}")
raf_label_counts = Counter(raf_ds["label"])
print(f"Label distribution of deanngkl/raf-db-7emotions: {raf_label_counts}")
print(f"ds[0]: {raf_ds[0]}")  # Print first sample to check label format