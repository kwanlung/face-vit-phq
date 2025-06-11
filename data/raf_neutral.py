from datasets import load_dataset, concatenate_datasets, Features, Sequence, Value, ClassLabel, Image
from collections import Counter
import numpy as np

# Load original datasets
raf_db = load_dataset("deanngkl/raf-db-7emotions", split="train")
affectnet = load_dataset("deanngkl/affectnet_no_contempt", split="train")

# Filter for neutral samples
neutral_affectnet = affectnet.filter(lambda x: x["label"] == 4)

# Add float32 bbox
def add_bbox(example):
    example["bbox"] = [np.float32(v) for v in [0.5, 0.5, 1.0, 1.0]]
    return example

neutral_affectnet = neutral_affectnet.map(add_bbox)

# Align schema using cast
neutral_affectnet = neutral_affectnet.remove_columns(
    [col for col in neutral_affectnet.column_names if col not in raf_db.column_names]
)
neutral_affectnet = neutral_affectnet.cast(raf_db.features)

# Concatenate
augmented_raf_db = concatenate_datasets([raf_db, neutral_affectnet])

# Display results
print("Original RAF-DB:", Counter(raf_db["label"]))
print("Neutral Added:", len(neutral_affectnet))
print("New Augmented Distribution:", Counter(augmented_raf_db["label"]))

# Save
augmented_raf_db.save_to_disk("data/local_data/raf-db-7emotions-neutral-added")
