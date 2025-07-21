# Facial Expression Recognition with ViT-Tiny

A deep learning project that fine-tunes a Vision Transformer (ViT-Tiny) model for 7-class facial emotion classification using cleaned versions of FER+, AffectNet, and RAF-DB datasets.

## 📌 Project Highlights

- 🔍 7-class emotion classification: `['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']`
- 🧠 Model: ViT-Tiny (`timm` implementation)
- 🎯 Achieved 82% validation accuracy
- 📚 Cleaned & uploaded datasets to Hugging Face Datasets
- 🧪 Integrated CutMix, cosine decay scheduler, and AMP for training

---

## 🗂️ Project Structure
```html
face_vit_phq/
├── configs/ 
├──── config.yaml         # YAML + Python config system
├── data/
├──── dataset.py          # Dataset preparation
├── model/ 
├──── vit_model.py        # ViT architecture, loss, forward pass
├── train.py              # Main training script
├── eval.py               # Evaluation script
├── app.py                # Run the model with simpel HTML
├── templates/
├──── index.html          # Simple template with Fast API
├── requirements.txt      # Dependencies
└── README.md
```

---

## Ⓜ️ Model
[Hugging Face Facial Expression Recognition ViT-Tiny](https://huggingface.co/deanngkl/vit-tiny-fer)

---

## 📊 Metrics
[Tensorboard Logs](https://huggingface.co/deanngkl/vit-tiny-fer/tensorboard)

---

## 📦 Datasets

| Dataset     | Link                                                                 | Notes                          |
|-------------|----------------------------------------------------------------------|--------------------------------|
| FER+        | [ferplus-7cls](https://huggingface.co/datasets/deanngkl/ferplus-7cls)         | Filtered to 7 basic emotions   |
| AffectNet   | [affectnet_no_contempt](https://huggingface.co/datasets/deanngkl/affectnet_no_contempt) | Removed 'contempt' class       |
| RAF-DB      | [raf-db-7emotions](https://huggingface.co/datasets/deanngkl/raf-db-7emotions)      | Added proper emotion labels    |

The total amount of datasets

```html
Loaded 75398 training samples from 3 sources
Loaded 8377 validation samples from 3 sources
Training-set distribution:
  0: 0 : 9738
  1: 1 : 3385
  2: 2 : 4313
  3: 3 : 18315
  4: 4 : 20987
  5: 5 : 9289
  6: 6 : 9371
Emotion batch torch.Size([64, 3, 224, 224])
```

---

## 🧪 Image Augmentation Strategy
Effective augmentation is key to improving model generalization, especially in emotion recognition tasks where expressions can vary widely due to lighting, angles, and facial features.

### 🔁 Transformations Used

- `Resize(224x224)`: Standardizes input for ViT models

- `RandomHorizontalFlip`: Encourages symmetry learning

- `RandomRotation(±10°)`: Adds rotation invariance

- `ColorJitter`: Simulates real-world lighting variance

- `Normalize`: Scales pixel values to `[-1, 1]` for ViT compatibility

<details>
<summary>📦 PyTorch Code</summary>

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```
</details>

### ✂️ CutMix Augmentation
To further boost performance and reduce overfitting, CutMix was applied during training:

- Combines two images by cutting and mixing patches

- Linearly blends labels based on area

- Helps the ViT backbone learn more robust, occlusion-resistant features

This proved particularly beneficial for minority classes like disgust and fear in RAF-DB and AffectNet.

> 💡 Insight: ViTs benefit greatly from mixed-structure inputs, making CutMix a powerful augmentation for patch-based embeddings.

---

## ⚙️ Environment & Compatibility
- Python >= 3.11
- PyTorch >= 2.2 (CUDA 12.1 recommended)  
  ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```
- Tested with GPU: NVIDIA RTX 3060 (16GB RAM)

---

## 🚀 How to Run

### 1. Clone project
```bash
git clone https://github.com/kwanlung/face-vit-phq.git
```

### 2. Setup Enviroment (For Windows)
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---
## Choose you want to train, eval or run the app
## Train the Model

```bash
python train.py
```

---

## Evaluate Model

```bash
python eval.py
```

## Run app
```bash
uvicorn app:app --host localhost --port 8000 --reload
```

---


| Dataset   | Val Accuracy | Backbone        | Notes                     |
| --------- | ------------ | --------------- | ------------------------- |
| FER+      | 82%        | ViT-Tiny (timm)    | CutMix + cosine scheduler |
| AffectNet | 82%           | ViT-Tiny        |         CutMix + cosine scheduler                  |
| RAF-DB    | 82%           | ViT-Tiny        |         CutMix + cosine scheduler                  |


---

📉 Detailed Evaluation Metrics
The model achieved an overall best validation accuracy of 82.2%, with the following precision, recall, and F1-scores for each class (Epoch 60):

| Emotion          | Precision | Recall    | F1-score  | Support |
| ---------------- | --------- | --------- | --------- | ------- |
| Anger            | 0.733     | 0.759     | 0.746     | 1085    |
| Disgust          | 0.685     | 0.532     | 0.599     | 393     |
| Fear             | 0.616     | 0.542     | 0.577     | 476     |
| Happiness        | 0.934     | 0.937     | 0.936     | 1971    |
| Neutral          | 0.895     | 0.918     | 0.906     | 2379    |
| Sadness          | 0.704     | 0.713     | 0.709     | 1047    |
| Surprise         | 0.775     | 0.794     | 0.784     | 1026    |
| **Macro Avg**    | **0.763** | **0.742** | **0.751** | 8377    |
| **Weighted Avg** | **0.819** | **0.822** | **0.820** | 8377    |

**Insights:**

- High performance on classes like Happiness and Neutral, indicating robust learning on frequent emotions.

- Moderate challenges with minority classes (Disgust, Fear), highlighting class imbalance.

- Good macro-average F1-score indicates balanced precision and recall across the emotion spectrum.
---

## 🔧 Hyperparameters (config.yaml)
| Parameter           | Value                  | Explanation & Tuning Insight                                                                                                                      |
| ------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`        | `vit_tiny_patch16_224` | Model backbone: Pretrained ViT-Tiny (`timm` implementation).                                                                                      |
| `optimizer`         | `AdamW`                | Optimizer: Adaptive optimizer balancing regularization and convergence speed.                                                                     |
| `learning_rate`     | `3e-4`                 | Initial learning rate. Cosine decay scheduler effectively decreased LR, enhancing model stability and preventing overfitting.                     |
| `batch_size`        | `64`                   | Optimal batch size considering GPU memory constraints (RTX 3060).                                                                                 |
| `num_epochs`        | `60`                   | Number of training epochs sufficient for convergence, as validation accuracy plateaued after \~50 epochs.                                         |
| `scheduler`         | `cosine`               | Cosine LR decay enhanced model performance by gradually reducing LR.                                                                              |
| `augmentation`      | `CutMix`               | Augmentation method significantly improved generalization, particularly for complex datasets like AffectNet and RAF-DB.                           |
| `mixed_precision`   | `true`                 | Enabled AMP (Automatic Mixed Precision) to improve computational efficiency and GPU utilization without compromising accuracy.                    |
| `early_stopping`    | `false`                | Training executed fully for 60 epochs; consider enabling early stopping in future to prevent potential overfitting and reduce unnecessary epochs. |
| `weighted_sampling` | `true`                 | Weighted sampling (`WeightedRandomSampler`) mitigated class imbalance, improving accuracy on minority classes.                                    |

---

## 🚦 API Usage Example
Run the FastAPI app locally and send a request:
```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "Content-Type: application/json" \
     -d '{"image_path": "sample.jpg"}'
```
**Example Response:**
```json
{
  "emotion": "happiness",
  "confidence": 0.95
}
```
---


**5. Troubleshooting Section:**
- Briefly highlight common issues users might face (e.g., GPU memory errors, version incompatibility).

**Example:**  
⚠️ Troubleshooting

- **CUDA Out of Memory Error**:  
  Consider lowering the batch size or disabling AMP.

- **Version Compatibility**:  
  Ensure CUDA version matches PyTorch’s requirements.

---

## 📌 Ethical Considerations
- Ensure data privacy and ethical compliance when deploying this model.
- Model predictions should support but not solely inform critical decisions related to mental health.

---


## 📈 Future Work
🔧 Train ViT-Base with mixed precision

🔬 Explore LoRA fine-tuning

🧠 Add regression head for PHQ score prediction

💬 Integrate with a depression detection pipeline

---

🤝 Contributing
Pull requests are welcome. Please create issues for feature requests or bugs.

---

## 📖 Citations
- Vision Transformer: Dosovitskiy et al. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- FER+: Barsoum et al. [Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution](https://arxiv.org/abs/1608.01041)

---

📄 License
MIT License

---

🙋‍♂️ Author
### Dean Ng Kwan Lung
Blog        : [Portfolio](https://kwanlung.github.io/)  
LinkedIn    : [LinkedIn](https://www.linkedin.com/in/deanng00/)  
GitHub      : [GitHub](https://github.com/kwanlung)  
Email       : kwanlung123@gmail.com