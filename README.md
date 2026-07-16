# AI-Generated Image Detector

A Python-based deep learning system that classifies whether an image is a **real photograph** or **AI-generated**, built on the CIFAKE dataset with a custom-trained CNN and a simple GUI for real-world testing.

## 🎯 Overview

With AI image generators becoming increasingly photorealistic, telling real photos apart from synthetic ones is getting harder — even for humans. This project trains a convolutional neural network to make that call automatically, and wraps it in a lightweight desktop GUI so anyone can drag in an image and get an instant prediction with a confidence score.

## ✨ Features

- **Binary classification**: REAL vs. AI-GENERATED
- **Confidence scoring**: every prediction comes with a percentage confidence
- **Low-confidence flagging**: predictions under a set threshold are marked "Low confidence – verify manually" instead of forcing a false sense of certainty
- **Simple GUI**: load any image and get a prediction without touching code
- **Trained on CIFAKE**: a large, labeled dataset of real and AI-generated images

## 🗂️ Dataset

Built on the **CIFAKE** dataset — labeled folders of real and AI-generated images, restructured and preprocessed for training:

```
dataset/
├── real/
│   ├── img1.jpg
│   └── ...
└── ai_generated/
    ├── img1.jpg
    └── ...
```

Images were resized and normalized for consistent model input, with folder structure and image size mismatches resolved during preprocessing.

## 🧠 Model

- Convolutional Neural Network (CNN) trained from scratch on the CIFAKE dataset
- Iteratively improved by fixing dataset folder mismatches, image size inconsistencies, and increasing training epochs for better convergence

## 📊 Results

| Class | Accuracy |
|---|---|
| FAKE (AI-generated) | 87.29% |
| REAL | 91.38% |

The model performs strongly overall, with most misclassifications occurring on **borderline, stock-photo-style images** — a known hard case, since modern AI generators are especially good at producing polished, corporate/office-style photography that closely mimics real stock photos.

## ⚠️ Limitations

- Roughly 1 in 8 fair test images may still be misclassified, particularly ambiguous/borderline ones
- The model is designed for **real photographs vs. AI-generated photographs** — testing it on illustrations, banners, or digital art is outside its intended scope and will produce unreliable results
- Low-confidence predictions (~50-55%) should be treated as uncertain, not wrong — this is the confidence flag working as intended

## 🚀 Getting Started

### Requirements
```bash
pip install tensorflow numpy pillow
```

### Usage
1. Clone the repository
2. Run the training script (or use the pre-trained model, if included)
3. Launch the GUI:
```bash
python gui.py
```
4. Select an image and view the prediction + confidence score

## 🔮 Future Improvements

- Expand training data with more stock-photo-style and borderline examples to close the confidence gap
- Add explainability (e.g., highlighting regions the model focused on)
- Package as a standalone executable for non-technical users
- Deploy as a web app for browser-based access

## 📄 License

This project is open for educational and portfolio use.

---
*Built as part of an ongoing exploration into practical applications of computer vision and deep learning.*
