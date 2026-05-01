"""
features.py — feature extraction and augmentation for the GTZAN genre
classification project.

Run independently:
    python features.py [--force]

Outputs (all in outputs/cache/):
    features.csv          — original track features
    augmented_features.csv — augmented training track features
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import librosa

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import (
    GENRES_DIR, CORRUPT, SR, DURATION, N_MFCC,
    FEAT_CACHE, AUG_CACHE,
    AUG_TYPES, N_AUGMENTATIONS, N_WORKERS,
    TEST_SIZE, RANDOM_STATE,
)
from utils import get_logger, timed, section

log = get_logger("features")


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_features_from_y(y, sr=SR, n_mfcc=N_MFCC):
    """Extract full feature vector from a pre-loaded audio array."""
    f = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        f[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        f[f"mfcc{i+1}_var"]  = np.var(mfcc[i])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f["chroma_stft_mean"] = np.mean(chroma)
    f["chroma_stft_var"]  = np.var(chroma)

    rms = librosa.feature.rms(y=y)
    f["rms_mean"] = np.mean(rms)
    f["rms_var"]  = np.var(rms)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    f["spectral_centroid_mean"] = np.mean(sc)
    f["spectral_centroid_var"]  = np.var(sc)

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f["spectral_bandwidth_mean"] = np.mean(sb)
    f["spectral_bandwidth_var"]  = np.var(sb)

    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    f["rolloff_mean"] = np.mean(roll)
    f["rolloff_var"]  = np.var(roll)

    zcr = librosa.feature.zero_crossing_rate(y)
    f["zcr_mean"] = np.mean(zcr)
    f["zcr_var"]  = np.var(zcr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    f["tempo"] = float(np.squeeze(tempo))

    mfcc_d = librosa.feature.delta(mfcc)
    for i in range(n_mfcc):
        f[f"mfcc{i+1}_delta_mean"] = np.mean(mfcc_d[i])
        f[f"mfcc{i+1}_delta_var"]  = np.var(mfcc_d[i])

    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    for i in range(n_mfcc):
        f[f"mfcc{i+1}_delta2_mean"] = np.mean(mfcc_d2[i])
        f[f"mfcc{i+1}_delta2_var"]  = np.var(mfcc_d2[i])

    sc2 = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(sc2.shape[0]):
        f[f"spec_contrast{i+1}_mean"] = np.mean(sc2[i])
        f[f"spec_contrast{i+1}_var"]  = np.var(sc2[i])

    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    for i in range(tonnetz.shape[0]):
        f[f"tonnetz{i+1}_mean"] = np.mean(tonnetz[i])
        f[f"tonnetz{i+1}_var"]  = np.var(tonnetz[i])

    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    f["chroma_cens_mean"] = np.mean(cens)
    f["chroma_cens_var"]  = np.var(cens)

    flat = librosa.feature.spectral_flatness(y=y)
    f["spectral_flatness_mean"] = np.mean(flat)
    f["spectral_flatness_var"]  = np.var(flat)

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    f["onset_strength_mean"] = np.mean(onset)
    f["onset_strength_var"]  = np.var(onset)
    f["onset_strength_max"]  = np.max(onset)

    return f


def extract_features(file_path, sr=SR):
    """Load audio from disk, then extract features."""
    y, sr = librosa.load(file_path, sr=sr, duration=DURATION)
    return extract_features_from_y(y, sr)


# ── Augmentation ──────────────────────────────────────────────────────────────

def apply_augmentation(y, aug_type, sr=SR):
    """Apply a single augmentation to waveform y in memory."""
    target_len = len(y)
    if aug_type == "time_stretch":
        y = librosa.effects.time_stretch(y, rate=np.random.choice([0.9, 1.1]))
        y = y[:target_len] if len(y) > target_len else np.pad(y, (0, target_len - len(y)))
    elif aug_type == "pitch_shift":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=int(np.random.choice([-2, 2])))
    elif aug_type == "noise":
        y = y + np.random.normal(0, 0.005 * np.std(y), len(y)).astype(y.dtype)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
    return y


# ── Parallel extraction ───────────────────────────────────────────────────────

def _worker_original(args):
    fpath, genre = args
    try:
        feats = extract_features(fpath)
        feats["label"]    = genre
        feats["filename"] = os.path.basename(fpath)
        return feats, None
    except Exception as e:
        return None, (fpath, str(e))


def _worker_augment(args):
    fpath, genre, fname, aug_idx, aug_type, seed = args
    try:
        np.random.seed(seed)
        y, _ = librosa.load(fpath, sr=SR, duration=DURATION)
        y_aug = apply_augmentation(y, aug_type)
        feats = extract_features_from_y(y_aug)
        feats["label"]    = genre
        feats["filename"] = f"{fname}_aug{aug_idx}_{aug_type}"
        return feats, None
    except Exception as e:
        return None, (fname, aug_type, str(e))


# ── Public API ────────────────────────────────────────────────────────────────

@timed(log)
def extract_all_features(all_files, force=False):
    """Extract features for all tracks. Uses cache unless force=True."""
    if not force and os.path.exists(FEAT_CACHE):
        log.info(f"Cache hit: {FEAT_CACHE} (use --force to recompute)")
        df = pd.read_csv(FEAT_CACHE)
        log.info(f"  {df.shape[0]} samples, {df.shape[1] - 2} features")
        return df

    section(log, "Feature Extraction — Original Tracks")
    log.info(f"Extracting features for {len(all_files)} tracks "
             f"using {N_WORKERS} processes …")

    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker_original, arg): arg for arg in all_files}
        for future in tqdm(as_completed(futures), total=len(all_files),
                           desc="Extracting", unit="track"):
            feats, err = future.result()
            if err:
                failed.append(err)
                tqdm.write(f"  ⚠ {os.path.basename(err[0])}: {err[1]}")
            else:
                rows.append(feats)

    log.info(f"Succeeded: {len(rows)}  |  Failed: {len(failed)}")
    df = pd.DataFrame(rows)
    df.to_csv(FEAT_CACHE, index=False)
    log.info(f"Saved → {FEAT_CACHE}  shape={df.shape}")
    return df


@timed(log)
def extract_augmented_features(df_train, force=False):
    """Extract augmented features for training tracks. Uses cache unless force=True."""
    if not force and os.path.exists(AUG_CACHE):
        log.info(f"Cache hit: {AUG_CACHE} (use --force to recompute)")
        df_aug = pd.read_csv(AUG_CACHE)
        log.info(f"  {df_aug.shape[0]} augmented samples")
        return df_aug

    section(log, "Feature Extraction — Augmented Training Tracks")
    np.random.seed(RANDOM_STATE)

    tasks = [
        (
            os.path.join(GENRES_DIR, row["label"], row["filename"]),
            row["label"],
            row["filename"],
            aug_idx,
            AUG_TYPES[aug_idx % len(AUG_TYPES)],
        )
        for _, row in df_train.iterrows()
        for aug_idx in range(N_AUGMENTATIONS)
    ]
    # Each process inherits the same RNG state at fork time, so give each
    # task a unique deterministic seed to avoid duplicate augmentations.
    task_seeds = np.random.randint(0, 2**31, size=len(tasks))
    tasks = [(*t, int(s)) for t, s in zip(tasks, task_seeds)]

    log.info(f"{len(df_train)} training tracks × {N_AUGMENTATIONS} = "
             f"{len(tasks)} augmented samples to extract …")

    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker_augment, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks),
                           desc="Augmenting", unit="track"):
            feats, err = future.result()
            if err:
                failed.append(err)
                tqdm.write(f"  ⚠ {err[0]} ({err[1]}): {err[2]}")
            else:
                rows.append(feats)

    log.info(f"Succeeded: {len(rows)}  |  Failed: {len(failed)}")
    df_aug = pd.DataFrame(rows)
    df_aug.to_csv(AUG_CACHE, index=False)
    log.info(f"Saved → {AUG_CACHE}  shape={df_aug.shape}")
    return df_aug


def collect_files():
    """Collect all valid audio file paths and their genres."""
    genres = sorted(os.listdir(GENRES_DIR))
    all_files = []
    for genre in genres:
        gdir = os.path.join(GENRES_DIR, genre)
        for fname in sorted(os.listdir(gdir)):
            if fname.endswith(".wav") and fname not in CORRUPT:
                all_files.append((os.path.join(gdir, fname), genre))
    return all_files


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force=False):
    section(log, "features.py — GTZAN Feature Pipeline")

    all_files = collect_files()
    log.info(f"Valid tracks: {len(all_files)}")

    df = extract_all_features(all_files, force=force)

    # Determine which tracks are training tracks (for augmentation)
    le_tmp = LabelEncoder()
    y_tmp = le_tmp.fit_transform(df["label"].values)
    train_idx, _ = train_test_split(
        np.arange(len(y_tmp)), test_size=TEST_SIZE,
        stratify=y_tmp, random_state=RANDOM_STATE
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    log.info(f"Training tracks (for augmentation): {len(df_train)}")

    df_aug = extract_augmented_features(df_train, force=force)

    section(log, "features.py complete")
    return df, df_aug


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTZAN feature extraction")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract features even if caches exist")
    args = parser.parse_args()
    main(force=args.force)
