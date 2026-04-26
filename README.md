# 🎵 CodeAlpha AI Music Generator

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.3-black?style=for-the-badge&logo=flask)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![music21](https://img.shields.io/badge/music21-MIDI-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> An AI-powered music generation system built with LSTM deep learning,
> trained on 292 classical piano pieces to compose unique melodies every time.

---

## 🌐 Live Demo

🔗 [![LinkedIn]https://www.linkedin.com/in/zainab-ilyas-559109349/]


---

## ✨ Features

- 🧠 **3-Layer LSTM Model** — 512 → 512 → 256 units with dropout regularization
- 🎹 **Classical Piano** — Trained on 292 MIDI classical piano compositions
- 🎼 **Unique Every Time** — Temperature sampling ensures every generation is original
- 🎨 **Creativity Slider** — Control music style from Structured to Experimental
- ▶️ **Browser Playback** — Play generated music directly in browser
- ⬇️ **Download** — Save generated music as WAV or MIDI
- 🌐 **Beautiful UI** — Dark theme with flying particles animation
- 🔄 **Restart** — Replay same composition from beginning

---

## 🏗️ Project Structure

```
codealpha-ai-music-generator/
├── app.py                      ← Flask backend (API + model)
├── requirements.txt            ← Python dependencies
├── README.md                   ← Project documentation
├── .gitignore                  ← Git ignore rules
├── static/
│   ├── style.css               ← UI styling
│   └── generated/              ← Generated music output
│       ├── output.wav          ← Generated WAV file
│       └── output.mid          ← Generated MIDI file
└── templates/
    └── index.html              ← Frontend UI
```

---

## 🧠 Model Architecture

```
Input Shape: (100 notes sequence, 1)
        ↓
┌─────────────────────────────────┐
│  LSTM Layer 1 — 512 units       │
│  return_sequences = True        │
└─────────────────────────────────┘
        ↓
    Dropout (0.3)
        ↓
┌─────────────────────────────────┐
│  LSTM Layer 2 — 512 units       │
│  return_sequences = True        │
└─────────────────────────────────┘
        ↓
    Dropout (0.3)
        ↓
┌─────────────────────────────────┐
│  LSTM Layer 3 — 256 units       │
│  return_sequences = False       │
└─────────────────────────────────┘
        ↓
    Dropout (0.3)
        ↓
┌─────────────────────────────────┐
│  Dense — 256 units (ReLU)       │
└─────────────────────────────────┘
        ↓
    Dropout (0.3)
        ↓
┌─────────────────────────────────┐
│  Dense — 222 units (Softmax)    │
└─────────────────────────────────┘
        ↓
  Generated Note / Chord
```

---

## 📊 Dataset Details

| Property | Details |
|---|---|
| **Source** | Kaggle — Classical Piano MIDI |
| **Total Files** | 292 MIDI files |
| **Genre** | Classical Piano |
| **Unique Pitches** | 222 |
| **Sequence Length** | 100 notes |
| **Training Platform** | Google Colab (T4 GPU) |
| **Training Epochs** | 100 |
| **Batch Size** | 64 |
| **Loss Function** | Categorical Crossentropy |
| **Optimizer** | Adam |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- Git

### Step 1 — Clone Repository

```bash
git clone https://github.com/zainabilyas205-source/Code-Alpha-AI-Music-Generator.git
cd codealpha-ai-music-generator
```

### Step 2 — Create Virtual Environment

```bash
py -3.10 -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add Model Files

Download these files from Google Drive and place in root folder:
- `model_weights.weights.h5`
- `vocab.pkl`
- `notes.pkl`
- `static/generated/output.wav`

### Step 5 — Run App

```bash
python app.py
```

### Step 6 — Open Browser

```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
flask==2.3.3
tensorflow==2.13.0
keras==2.13.1
music21==9.1.0
numpy==1.24.3
midi2audio==0.1.1
```

---

## 🔄 How It Works

```
Step 1 — Data Collection
         292 Classical Piano MIDI files (Kaggle)
                    ↓
Step 2 — Preprocessing (music21)
         Parse notes and chords from MIDI
         Build sequences of 100 notes each
                    ↓
Step 3 — Model Training (Google Colab GPU)
         3-Layer LSTM architecture
         100 epochs — Categorical Crossentropy
                    ↓
Step 4 — Music Generation
         Random seed sequence selected
         Temperature sampling applied
         100 new notes generated
                    ↓
Step 5 — MIDI → WAV Conversion
         FluidSynth + SoundFont
                    ↓
Step 6 — Web Interface
         Browser playback + Download
```

---

## 🎨 Temperature Control

| Temperature | Music Style |
|---|---|
| **0.3** | Very structured, repetitive, safe |
| **0.5** | Structured with some variation |
| **0.8** | Balanced — recommended ✅ |
| **1.0** | Creative and varied |
| **1.2** | Very experimental, random |

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.10 | Core language |
| **TensorFlow** | 2.x | Deep learning framework |
| **Keras** | 2.x | LSTM model building |
| **music21** | 9.x | MIDI parsing and generation |
| **Flask** | 2.3 | Web server and API |
| **NumPy** | 1.24 | Numerical computations |
| **Google Colab** | — | GPU model training |
| **HTML/CSS/JS** | — | Frontend UI |

---

## 🎯 Results

| Metric | Value |
|---|---|
| **Vocabulary Size** | 222 unique pitches |
| **Training Files** | 292 MIDI files |
| **Sequence Length** | 100 notes |
| **Generation Speed** | ~1-2 minutes |
| **Output Formats** | WAV + MIDI |
| **Music Style** | Classical Piano |

---

## ⚠️ Important Notes

- `model_weights.weights.h5` is not included in repo due to file size
- `vocab.pkl` and `notes.pkl` are not included — train the model yourself
- Place `output.wav` in `static/generated/` for browser playback
- Use Google Colab for training — local training is very slow

---

## 🏆 Training on Google Colab

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Train model
history = model.fit(
    X, y,
    epochs=100,
    batch_size=64,
    callbacks=[checkpoint, early_stop]
)
```

---

## 👩‍💻 Author

**YOUR NAME**

[![GitHub]https://github.com/zainabilyas205-source/Code-Alpha-AI-Music-Generator]
[![LinkedIn]https://www.linkedin.com/in/zainab-ilyas-559109349/]

---

## 🏆 Acknowledgements

- **CodeAlpha** — Internship project opportunity
- **Kaggle** — Classical Piano MIDI dataset
- **Google Colab** — Free GPU training environment
- **music21 (MIT)** — Music analysis library

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2026 Zainab Ilyas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

⭐ **If you found this project helpful, please give it a star!** ⭐

---

*Built with ❤️ using Python 3.10 • TensorFlow • Flask • music21*
