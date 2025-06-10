from datasets import load_from_disk

# Load the filtered dataset
dataset_path = "face_vit_phq/data/local_data/raf-db-7emotions-neutral-added"
dataset = load_from_disk(dataset_path)

# Push to Hugging Face Hub
dataset.push_to_hub(
    repo_id="deanngkl/raf-db-7emotions",
    private=False,  # Set to True if you want the dataset to be private
)
print("Dataset uploaded successfully!")