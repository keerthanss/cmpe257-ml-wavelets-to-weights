"""
eda.py — Exploratory Data Analysis for the GTZAN genre classification project.

Saves all figures to outputs/plots/eda/. Run independently:
    python eda.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

import librosa
import librosa.display

from config import (
    GENRES_DIR, CORRUPT, SR, DURATION,
    PLOTS_EDA, FEAT_CACHE,
)
from utils import get_logger, timed, section

log = get_logger("eda")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130

PALETTE = sns.color_palette("tab10", 10)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_genres():
    genres = sorted(os.listdir(GENRES_DIR))
    log.info(f"Genres found: {genres}")
    return genres


def savefig(name: str):
    path = os.path.join(PLOTS_EDA, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path}")


# ── 1. Class Balance ──────────────────────────────────────────────────────────

@timed(log)
def plot_class_balance(genres):
    section(log, "1. Class Balance")
    counts = {}
    for genre in genres:
        gdir = os.path.join(GENRES_DIR, genre)
        n = sum(1 for f in os.listdir(gdir)
                if f.endswith(".wav") and f not in CORRUPT)
        counts[genre] = n
        log.debug(f"  {genre}: {n} tracks")

    log.info(f"Total valid tracks: {sum(counts.values())}")

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.bar(counts.keys(), counts.values(),
                  color=PALETTE, edgecolor="white")
    ax.set_title("Track count per genre (corrupt file excluded)")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 115)
    for bar, (genre, cnt) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", fontsize=9)
    plt.tight_layout()
    savefig("01_class_balance.png")


# ── 2. Waveforms ──────────────────────────────────────────────────────────────

def _load_waveform(genre):
    path = os.path.join(GENRES_DIR, genre, f"{genre}.00000.wav")
    y, sr = librosa.load(path, sr=SR, duration=DURATION)
    return genre, y, sr


@timed(log)
def plot_waveforms(genres):
    section(log, "2. Waveforms")
    log.info("Loading waveforms in parallel …")

    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_load_waveform, genres))

    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, (genre, y, sr) in enumerate(results):
        librosa.display.waveshow(y, sr=sr, ax=axes[idx], color=PALETTE[idx])
        axes[idx].set_title(genre.capitalize(), fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("")
    fig.suptitle("Waveforms — one representative track per genre",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    savefig("02_waveforms.png")


# ── 3. Mel Spectrograms ───────────────────────────────────────────────────────

def _load_mel(genre):
    path = os.path.join(GENRES_DIR, genre, f"{genre}.00000.wav")
    y, sr = librosa.load(path, sr=SR, duration=DURATION)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    return genre, S_db, sr


@timed(log)
def plot_mel_spectrograms(genres):
    section(log, "3. Mel Spectrograms")
    log.info("Computing mel spectrograms in parallel …")

    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_load_mel, genres))

    fig, axes = plt.subplots(5, 2, figsize=(14, 14))
    axes = axes.flatten()
    for idx, (genre, S_db, sr) in enumerate(results):
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                       y_axis="mel", ax=axes[idx], cmap="magma")
        axes[idx].set_title(genre.capitalize(), fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("")
        fig.colorbar(img, ax=axes[idx], format="%+2.0f dB")
    fig.suptitle("Mel Spectrograms — one representative track per genre",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    savefig("03_mel_spectrograms.png")


# ── 4. ZCR / RMS distributions ───────────────────────────────────────────────

def _compute_zcr_rms(args):
    fpath, genre = args
    y, _ = librosa.load(fpath, sr=SR, duration=DURATION)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return zcr, rms, genre


@timed(log)
def plot_zcr_rms(genres):
    section(log, "4. ZCR / RMS Distributions")
    np.random.seed(42)

    tasks = []
    for genre in genres:
        gdir = os.path.join(GENRES_DIR, genre)
        files = [f for f in sorted(os.listdir(gdir))
                 if f.endswith(".wav") and f not in CORRUPT]
        sampled = np.random.choice(files, size=min(30, len(files)), replace=False)
        for fname in sampled:
            tasks.append((os.path.join(gdir, fname), genre))

    log.info(f"Loading {len(tasks)} tracks for ZCR/RMS scan …")
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_compute_zcr_rms, tasks))

    zcr_data, rms_data, labels = zip(*results)
    scan_df = pd.DataFrame({"genre": labels, "ZCR_mean": zcr_data,
                             "RMS_mean": rms_data})

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.boxplot(data=scan_df, x="genre", y="ZCR_mean",
                order=genres, palette="muted", ax=axes[0])
    axes[0].set_title("Zero-Crossing Rate by Genre")
    axes[0].tick_params(axis="x", rotation=30)
    sns.boxplot(data=scan_df, x="genre", y="RMS_mean",
                order=genres, palette="muted", ax=axes[1])
    axes[1].set_title("RMS Energy by Genre")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    savefig("04_zcr_rms.png")


# ── 5. Feature Distributions (from cache) ────────────────────────────────────

@timed(log)
def plot_feature_distributions(genres):
    section(log, "5. Feature Distributions")
    if not os.path.exists(FEAT_CACHE):
        log.warning(f"Feature cache not found at {FEAT_CACHE} — run features.py first.")
        return

    df = pd.read_csv(FEAT_CACHE)
    log.info(f"Loaded feature cache: {df.shape}")

    showcase = ["spec_contrast1_mean", "tonnetz3_mean",
                "spectral_flatness_mean", "onset_strength_mean",
                "mfcc1_mean", "chroma_stft_mean"]
    showcase = [c for c in showcase if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, showcase):
        sns.boxplot(data=df, x="label", y=feat, order=genres,
                    palette="muted", ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    plt.suptitle("Per-genre feature distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("05_feature_distributions.png")


# ── 6. Correlation Heatmap ────────────────────────────────────────────────────

@timed(log)
def plot_correlation(genres):
    section(log, "6. Correlation Heatmap")
    if not os.path.exists(FEAT_CACHE):
        log.warning(f"Feature cache not found — run features.py first.")
        return

    df = pd.read_csv(FEAT_CACHE)
    feat_cols = [c for c in df.columns if c not in ("label", "filename")]
    mean_cols = [c for c in feat_cols if c.endswith("_mean") or c == "tempo"]
    corr = df[mean_cols].corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    high = (corr.where(~mask).stack().reset_index()
                .rename(columns={"level_0": "a", "level_1": "b", 0: "r"})
                .query("r > 0.90 and a != b")
                .sort_values("r", ascending=False))
    log.info(f"Feature pairs |r| > 0.90: {len(high)}")
    if not high.empty:
        log.info(f"Top pair: {high.iloc[0]['a']} ↔ {high.iloc[0]['b']} "
                 f"(r={high.iloc[0]['r']:.3f})")

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0.5,
                linewidths=0, ax=ax, cbar_kws={"label": "|Pearson r|"})
    ax.set_title("Absolute Pearson Correlation — mean features", fontsize=13)
    plt.tight_layout()
    savefig("06_correlation_heatmap.png")


# ── 7. t-SNE ─────────────────────────────────────────────────────────────────

@timed(log)
def plot_tsne(genres):
    section(log, "7. t-SNE")
    if not os.path.exists(FEAT_CACHE):
        log.warning(f"Feature cache not found — run features.py first.")
        return

    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(FEAT_CACHE)
    feat_cols = [c for c in df.columns if c not in ("label", "filename")]
    X = StandardScaler().fit_transform(df[feat_cols].values)

    log.info("Running t-SNE (this may take a minute) …")
    Z = TSNE(n_components=2, perplexity=30, learning_rate=200,
             max_iter=1000, random_state=42).fit_transform(X)

    genre_to_color = {g: PALETTE[i] for i, g in enumerate(genres)}
    fig, ax = plt.subplots(figsize=(10, 8))
    for genre in genres:
        mask = df["label"].values == genre
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   label=genre.capitalize(), alpha=0.7, s=25,
                   color=genre_to_color[genre])
    ax.legend(loc="upper right", markerscale=1.5)
    ax.set_title("t-SNE of full feature vector (n=999 tracks)", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    savefig("07_tsne.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    section(log, "EDA — GTZAN Genre Classification")
    genres = get_genres()

    plot_class_balance(genres)
    plot_waveforms(genres)
    plot_mel_spectrograms(genres)
    plot_zcr_rms(genres)
    plot_feature_distributions(genres)
    plot_correlation(genres)
    plot_tsne(genres)

    section(log, "EDA complete — all plots saved to outputs/plots/eda/")


if __name__ == "__main__":
    main()
