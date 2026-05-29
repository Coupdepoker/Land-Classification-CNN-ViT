# 🛰️ Land Classification with CNN-ViT Hybrid Architecture

> Satellite image classification using a hybrid CNN + Vision Transformer (ViT) model — built and evaluated in both **Keras/TensorFlow** and **PyTorch**.

---

## 📌 Project Overview

This project implements an end-to-end deep learning pipeline to classify satellite images into two categories: **agricultural land** vs **non-agricultural land**.

The core innovation is a **CNN-ViT hybrid architecture** that combines:
- **CNNs** — to extract local spatial features (edges, textures, patterns)
- **Vision Transformers (ViT)** — to capture global spatial dependencies via self-attention

The entire pipeline was implemented in parallel across two major frameworks (Keras and PyTorch), with a final comparative evaluation of both.

---

## 🗂️ Project Structure

```
├── Module 1 — Data Loading & Augmentation
│   ├── Compare_Memory-Based_Versus_Generator-Based_Data_Loading.ipynb
│   ├── Lab_M1L2_Data_Loading_and_Augmentation_Using_Keras.ipynb
│   └── Lab_M1L3_Data_Loading_and_Augmentation_Using_PyTorch.ipynb
│
├── Module 2 — CNN Classifiers
│   ├── Lab_M2L1_Train_and_Evaluate_a_Keras-Based_Classifier.ipynb
│   ├── Lab_M2L2_Implement_and_Test_a_PyTorch-Based_Classifier.ipynb
│   └── Lab_M2L3_Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb
│
├── Module 3 — Vision Transformers
│   ├── Lab_M3L1_Vision_Transformers_in_Keras.ipynb
│   └── Lab_M3L2_Vision_Transformers_in_PyTorch.ipynb
│
└── Module 4 — Final Integration & Evaluation
    └── lab_M4L1_Land_Classification_CNN-ViT_Integration_Evaluation.ipynb
```

---

## 🧠 Architecture

The CNN-ViT hybrid model follows this pipeline:

```
Satellite Image
      │
      ▼
CNN Backbone  ──── local feature extraction (edges, textures)
      │
      ▼
Feature Maps → Patch Tokens
      │
      ▼
Vision Transformer Encoder  ──── global spatial attention
      │
      ▼
Classification Head  ──── agricultural / non-agricultural
```

---

## 🔬 What's Covered

### Module 1 — Data Engineering
- Memory-based vs generator-based data loading (trade-offs in RAM, speed, scalability)
- Custom `Dataset` class from scratch in PyTorch (`torch.utils.data.Dataset`)
- `ImageFolder` utility and `DataLoader` with batching, shuffling, and parallel loading
- Keras `image_dataset_from_directory` with `tf.data` API (`.map()`, `.cache()`, `.prefetch()`)
- Image augmentation pipelines (flips, rotations, color jitter)

### Module 2 — CNN Baselines
- Custom CNN architectures in both Keras and PyTorch
- Training loops, callbacks, early stopping
- Evaluation: accuracy, precision, recall, F1-score
- Framework comparison on identical tasks

### Module 3 — Vision Transformers
- ViT architecture from scratch: patch embedding, positional encoding, multi-head self-attention, MLP blocks
- CNN-ViT hybrid: CNN as feature extractor → ViT encoder for global context
- Implemented in both Keras (TensorFlow backend) and PyTorch

### Module 4 — Final Evaluation
- Loading and benchmarking pre-trained CNN-ViT models from both frameworks
- Cross-framework consistency check
- Quantitative evaluation: accuracy, precision, recall, F1-score
- Practical inference workflow on new satellite images

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras, PyTorch, torchvision |
| Data | NumPy, PIL, Matplotlib |
| Evaluation | scikit-learn (classification report, confusion matrix) |
| Dataset | Satellite imagery — agricultural vs non-agricultural land |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Coupdepoker/land-classification-cnn-vit.git
cd land-classification-cnn-vit

# Run notebooks in order (Module 1 → 4)
jupyter notebook
```

---

## 📊 Key Results

- CNN-ViT hybrid outperforms standalone CNN baselines on the satellite classification task
- The self-attention mechanism allows the model to capture long-range spatial dependencies missed by convolutions alone
- Both Keras and PyTorch implementations yield consistent, comparable results
- Evaluation metrics include accuracy, precision, recall, and F1-score across both frameworks

---

## 💡 Key Takeaways

- **CNNs excel at local feature extraction** but miss global context
- **Vision Transformers capture global dependencies** but require more data and compute
- **Hybrid CNN-ViT architectures** get the best of both worlds
- Generator-based data loading is significantly more scalable than memory-based loading for large image datasets

---
