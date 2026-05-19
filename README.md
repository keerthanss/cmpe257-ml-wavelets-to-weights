# From Wavelets to Weights: Music Genre Classification

A music genre classification system built on the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data), classifying audio tracks into 10 genres using signal processing features and classical machine learning models. The trained model is deployed as an interactive web app.

**Live App:** https://cmpe257-sjsu-wavelets-to-weights.streamlit.app/

---

## Overview

Music genre classification is a core task in music information retrieval, with applications in content-based recommendation and automated library organisation. This project extracts 166 audio features from raw `.wav` files using `librosa`, applies dimensionality reduction, trains five classical ML models, and deploys the best-performing model via Streamlit.

**Best result:** SVM (RBF kernel) with SelectKBest features — **85.5% test accuracy**.

---

## Dataset

- **Source:** GTZAN dataset (1000 tracks, 10 genres, 30s each, 22050Hz Mono 16-bit `.wav`)
- **Genres:** blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock
- **Effective size:** 999 tracks (`jazz.00054.wav` excluded due to format error)

---

## Pipeline

### Feature Engineering (`features.py`)
166 features extracted per track using `librosa`:
- MFCCs, MFCC Δ, MFCC ΔΔ (mean + variance) — 120 features
- Spectral centroid, rolloff, bandwidth, zero-crossing rate, chroma STFT, RMS, tempo — 13 features
- Spectral contrast — 14 features
- Tonnetz — 12 features
- Chroma CENS — 2 features
- Spectral flatness, onset strength — 5 features

Features are cached to CSV for fast subsequent runs.

### Dimensionality Reduction (`reduction.py`)
Three strategies evaluated:
- **PCA** — 95% variance threshold → 86 components
- **LDA** — reduces to 9 features (classes − 1)
- **SelectKBest** — ANOVA F-statistic, k=80

### Preprocessing
- Label encoding, feature standardisation (zero mean, unit variance)
- Stratified 80/20 train/test split
- Stratified 5-fold cross-validation on train split

### Data Augmentation
Training set tripled via:
- Time stretch (rates 0.9 and 1.1)
- Pitch shift (±2 semitones)
- Additive Gaussian noise (std = 0.5%)

Augmented features are precomputed and cached. Augmentation had negligible effect on final model performance.

### Models (`cross_validation.py`, `main.py`)
40 experiments total (5 models × 4 feature spaces × 2 augmentation settings):
- Gaussian Naive Bayes
- k-Nearest Neighbours
- Support Vector Machines
- Random Forest
- Gradient Boosted Trees (XGBoost)

**Best model:** SVM (RBF) + SelectKBest, no augmentation — CV accuracy 78.3% ± 2.2%, test accuracy **85.5%**.

### Web App (`app.py`)
Streamlit app supporting real-time inference. Upload an MP3 file; the app splits it into five 30-second segments, runs inference on each, and returns the genre by majority vote.

---

## Repository Structure

```
.
├── features.py          # Feature extraction using librosa
├── reduction.py         # PCA, LDA, SelectKBest
├── cross_validation.py  # Stratified k-fold CV across all experiments
├── main.py              # Entry point for training
├── app.py               # Streamlit web app
├── config.py            # Shared configuration
├── eda.py               # Exploratory data analysis plots
├── utils.py             # Helper utilities
├── models/              # Model helper scripts
├── outputs/             # Figures and experiment outputs
├── experiments.out      # Raw experiment results log
└── requirements.txt
```

---

## Data Download

The dataset is available on Kaggle at:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

The code expects the dataset to be placed **one level above the repo root**, at `../dataset/Data/genres_original/`. Your directory structure should look like:

```
parent_folder/
├── cmpe257-ml-wavelets-to-weights/   ← repo
└── dataset/
    └── Data/
        └── genres_original/
            ├── blues/
            ├── classical/
            └── ...
```
For alternate configurations, make the required changes in `config.py`.

**Option 1 — Manual download:**
1. Sign in to Kaggle and visit the link above
2. Click **Download** to get the zip file
3. Extract it

**Option 2 — Kaggle API:**
```bash
cd ..   # move one level above the repo
mkdir -p dataset/Data
cd dataset/Data
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip
```

> Requires a Kaggle account and `kaggle.json` API token placed at `~/.kaggle/kaggle.json`. See [Kaggle API docs](https://github.com/Kaggle/kaggle-api) for setup instructions.

---

## Installation

```bash
git clone https://github.com/keerthanss/cmpe257-ml-wavelets-to-weights.git
cd cmpe257-ml-wavelets-to-weights
pip install -r requirements.txt
```

> XGBoost is optional. Uncomment the `xgboost` line in `requirements.txt` if needed.

---

## Usage

**Train models:**
```bash
python main.py
```

**Run the web app locally:**
```bash
streamlit run app.py
```

---

## Dependencies

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| streamlit | 1.57.0 |
| tqdm | 4.67.3 |

---

## Course

CMPE 257 — Machine Learning, San José State University (Spring 2026)
