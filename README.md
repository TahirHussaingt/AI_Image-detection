# 🔍 AI Image Detector

A desktop application that detects whether an image is a **real photograph** or a **photorealistic AI-generated image**, built with TensorFlow and a simple Tkinter GUI.

## Overview

This project uses transfer learning on top of **MobileNetV2** (pretrained on ImageNet) to classify images as REAL or FAKE (AI-generated). A lightweight custom classification head was trained on top of the frozen pretrained base, allowing the model to leverage deep, general-purpose visual understanding while only needing to learn the specific task of spotting AI-generated content.

## Features

- 🖼️ Upload any image via file browser or drag-and-drop
- ⚡ Instant REAL / FAKE prediction with confidence percentage
- 🎨 Color-coded confidence levels (High / Moderate / Low – verify manually)
- 📊 Visual confidence bar
- 📜 Running history of the last 5 predictions
- 🛡️ Graceful handling of corrupted, empty, or unsupported files

## Model Details

| | |
|---|---|
| Base architecture | MobileNetV2 (ImageNet pretrained, frozen) |
| Input size | 224 × 224 × 3 |
| Custom head | GlobalAveragePooling → Dropout(0.3) → Dense(1, sigmoid) |
| Trainable parameters | 1,281 |
| Training data | 15,000 images (7,500 real / 7,500 AI-generated), sourced from a Kaggle "AI vs. Human-Generated Images" dataset (Shutterstock photos vs. DALL-E generations) |
| Epochs | 5 |
| **Overall test accuracy** | **89.33%** |
| Accuracy on REAL images | 91.38% |
| Accuracy on FAKE images | 87.29% |

## Project Structure

```
Image Detection/
├── dataset_v2/
│   ├── train/
│   │   ├── REAL/
│   │   └── FAKE/
│   └── test/
│       ├── REAL/
│       └── FAKE/
├── train_model.py        # Builds and trains the model
├── evaluate_model.py      # Evaluates model on the full test set
├── predict.py              # Command-line single-image prediction
├── gui.py                    # Desktop GUI application
├── prepare_dataset.py    # Builds dataset_v2 from raw Kaggle CSV + images
└── image_detector_model.keras   # Saved trained model
```

## Dataset

This project uses the **[AI vs. Human-Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)** dataset from Kaggle (Shutterstock real photos vs. DALL-E generated images).

> ⚠️ The dataset (~10GB) is **not included** in this repository due to size. To reproduce training:
> 1. Download the dataset from the Kaggle link above
> 2. Place `train.csv`, `test.csv`, `train_data/`, and `test_data_v2/` in the project root
> 3. Run `prepare_dataset.py` to build the balanced `dataset_v2/` folder used for training

## How to Run

### Option A: Just use the pretrained model (recommended)
The trained model (`image_detector_model.keras`) is included in this repo — no dataset download needed.

1. Install dependencies:
   ```bash
   pip install tensorflow pillow pillow-heif tkinterdnd2 pandas
   ```
2. Launch the GUI:
   ```bash
   python gui.py
   ```
3. Click **Choose Image** (or drag and drop a photo onto the app) to get a prediction.

### Option B: Retrain from scratch
1. Download the dataset (see [Dataset](#dataset) above)
2. Run `python prepare_dataset.py`
3. Run `python train_model.py`
4. Run `python evaluate_model.py` to verify accuracy
5. Run `python gui.py` to use your newly trained model

## Limitations

- Trained specifically on **photorealistic** real vs. AI images — not designed for illustrations, graphic design, infographics, or stylized artwork.
- No AI image detector achieves 100% accuracy; this model performs at ~89%, in line with industry-standard detection tools.
- Best suited for everyday photography-style images (people, objects, nature, scenes).

## Future Improvements

- Fine-tune deeper MobileNetV2 layers for higher accuracy
- Expand training data across more AI generators (Midjourney, Stable Diffusion, etc.)
- Add batch-processing support for multiple images at once
