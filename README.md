# 🎵 SoundLens — AI Music Genre Classifier

> Deep Learning project that classifies music genres from audio files using CNN on Mel Spectrograms.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Librosa](https://img.shields.io/badge/Librosa-0.10-orange) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-purple)

## 🎯 What It Does

Upload any audio file → AI analyzes the **Mel Spectrogram** → Predicts music genre with confidence scores + visualizations.

**Supported Genres:** Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock

## 🧠 How It Works (Tech Stack)

```
Audio File
    ↓
Librosa (Audio Loading)
    ↓
Feature Extraction:
  • Mel Spectrogram
  • MFCCs (13 coefficients)
  • Chroma Features
  • Spectral Centroid / Rolloff / Bandwidth
  • Zero Crossing Rate
  • Tempo (BPM)
    ↓
CNN Model (trained on GTZAN dataset)
    ↓
Genre Prediction + Confidence Scores
    ↓
Flask API → Beautiful Web UI
```

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/HetGhadiya/soundlens.git
cd soundlens
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

### 5. Open browser
```
http://localhost:5000
```

## 📁 Project Structure

```
soundlens/
│
├── app.py                  # Flask backend + feature extraction
├── requirements.txt        # Python dependencies
├── README.md
│
├── templates/
│   └── index.html          # Frontend UI
│
├── model/
│   └── (place your trained model here)
│
└── uploads/                # Temp folder (auto-created)
```

## 🏋️ Training Your Own Model (Optional)

1. Download GTZAN Dataset from Kaggle
2. Extract features using `librosa`
3. Train CNN on mel spectrograms:

```python
# Quick training script
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load GTZAN dataset and extract mel spectrograms
# Train CNN model
# Save as model/genre_model.h5
```

4. Load in `app.py`:
```python
model = tf.keras.models.load_model('model/genre_model.h5')
```

## 📊 Features Extracted

| Feature | Description |
|---|---|
| MFCCs | Mel Frequency Cepstral Coefficients (timbre) |
| Chroma | Pitch class distribution |
| Spectral Centroid | Brightness of sound |
| Spectral Rolloff | Shape of spectrum |
| Zero Crossing Rate | Noisiness of signal |
| Tempo | BPM of the track |

## 🎓 Dataset

- **GTZAN Genre Collection** — 1000 audio tracks, 10 genres, 30 seconds each
- Download: [Kaggle GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## 👨‍💻 Built By

**Het** — B.Tech AI & Data Science, ADIT (CVMU)

---

*Built for placement portfolio — demonstrates CNN + Audio Processing + Flask deployment*
