# Vision Transformer for Facial Expression-based PHQ-8/9 Regression

A deep learning project that fine-tunes a Vision Transformer (ViT-Tiny) model for 7-class facial emotion classification using cleaned versions of FER+, AffectNet, and RAF-DB datasets.

## 📌 Project Highlights

- 🔍 7-class emotion classification: `['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']`
- 🧠 Model: ViT-Tiny (`timm` implementation)
- 🎯 Achieved 81.4% validation accuracy
- 📚 Cleaned & uploaded datasets to Hugging Face Datasets
- 🧪 Integrated CutMix, cosine decay scheduler, and AMP for training

---

## 🗂️ Project Structure
```html
face_vit_phq/
├── configs/ # YAML + Python config system
├── data/ # Custom dataset loading, splits
├── model/ # ViT architecture, loss, forward pass
├── train.py # Main training script
├── eval.py # Evaluation script
├── requirements.txt # Dependencies
└── README.md
```

---

## 📦 Datasets

| Dataset     | Link                                                                 | Notes                          |
|-------------|----------------------------------------------------------------------|--------------------------------|
| FER+        | [Hugging Face](https://huggingface.co/datasets/deanngkl/ferplus-7cls)         | Filtered to 7 basic emotions   |
| AffectNet   | [Hugging Face](https://huggingface.co/datasets/deanngkl/affectnet_no_contempt) | Removed 'contempt' class       |
| RAF-DB      | [Hugging Face](https://huggingface.co/datasets/deanngkl/raf-db-7emotions)      | Added proper emotion labels    |

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

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py --config configs/config.yaml
```

---

## Evaluate Model

```bash
python eval.py --ckpt_path outputs/best_model.pth
```


| Dataset   | Val Accuracy | Backbone        | Notes                     |
| --------- | ------------ | --------------- | ------------------------- |
| FER+      | 81.4%        | ViT-Tiny (timm) | CutMix + cosine scheduler |
| AffectNet | TBD          | ViT-Tiny        |                           |
| RAF-DB    | TBD          | ViT-Tiny        |                           |


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

📄 License
MIT License

---

🙋‍♂️ Author
### Dean Ng Kwan Lung
Blog        : [Portfolio](https://kwanlung.github.io/)  
LinkedIn    : [LinkedIn](https://www.linkedin.com/in/deanng00/)  
GitHub      : [GitHub](https://github.com/kwanlung)  
Email       : kwanlung123@gmail.com