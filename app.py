"""
app.py — Streamlit web app for GTZAN music genre classification.

Run from the scripts/ directory:
    streamlit run app.py
"""

import sys
import os
import warnings
import tempfile
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure scripts/ is on sys.path so sibling modules import cleanly
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from config import SR, N_MFCC, FEAT_CACHE
from features import extract_features_from_y

# ── Constants ─────────────────────────────────────────────────────────────────

SCRIPTS_DIR  = Path(__file__).parent
MODEL_DIR    = (
    SCRIPTS_DIR / "outputs" / "runs" / "2026-05-01_00-19-47_svm" / "models"
)
MODEL_TAG    = "svm_rbf_skb_no-aug"
MODEL_PATH   = MODEL_DIR / f"{MODEL_TAG}_final.joblib"
SCALER_PATH  = MODEL_DIR / f"{MODEL_TAG}_scaler.joblib"
REDUCER_PATH = MODEL_DIR / f"{MODEL_TAG}_reducer.joblib"

N_SEGMENTS   = 5
SEG_DURATION = 30                    # seconds per segment (matches training clips)
SEG_SAMPLES  = SEG_DURATION * SR     # 661,500 samples at 22,050 Hz

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

GENRE_EMOJI = {
    "blues":     "🎸",
    "classical": "🎻",
    "country":   "🤠",
    "disco":     "🕺",
    "hiphop":    "🎤",
    "jazz":      "🎷",
    "metal":     "🤘",
    "pop":       "🎵",
    "reggae":    "🌴",
    "rock":      "⚡",
}

# Pastel background + border per genre
GENRE_COLORS = {
    "blues":     ("#dbeafe", "#93c5fd"),
    "classical": ("#fce7f3", "#f9a8d4"),
    "country":   ("#fef9c3", "#fde047"),
    "disco":     ("#ede9fe", "#c4b5fd"),
    "hiphop":    ("#fee2e2", "#fca5a5"),
    "jazz":      ("#d1fae5", "#6ee7b7"),
    "metal":     ("#f1f5f9", "#cbd5e1"),
    "pop":       ("#ffedd5", "#fdba74"),
    "reggae":    ("#dcfce7", "#86efac"),
    "rock":      ("#fdf4ff", "#e879f9"),
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GTZAN Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Step cards (How It Works) ── */
.step-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 14px;
    padding: 22px 14px 18px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    height: 100%;
    transition: box-shadow 0.2s;
}
.step-card:hover { box-shadow: 0 4px 18px rgba(79,70,229,0.12); }
.step-icon  { font-size: 2.1rem; margin-bottom: 9px; }
.step-title { font-weight: 700; font-size: 0.95rem; color: #4f46e5; margin-bottom: 5px; }
.step-desc  { font-size: 0.82rem; color: #6b7280; line-height: 1.45; }

/* ── Segment result cards ── */
.seg-card {
    border-radius: 12px;
    padding: 16px 10px 14px;
    text-align: center;
    border: 2px solid;
    margin-bottom: 2px;
}
.seg-num   { font-size: 0.72rem; font-weight: 700; color: #6b7280;
             text-transform: uppercase; letter-spacing: 0.05em; }
.seg-time  { font-size: 0.70rem; color: #9ca3af; margin: 3px 0 8px; }
.seg-emoji { font-size: 1.8rem; display: block; margin-bottom: 5px; }
.seg-genre { font-size: 1.05rem; font-weight: 700; color: #1f2937; }

/* ── Winner banner ── */
.winner-banner {
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
    border: 2px solid #a5b4fc;
    border-radius: 18px;
    padding: 32px 24px;
    text-align: center;
    margin-top: 12px;
}
.winner-label {
    font-size: 0.85rem; font-weight: 600; color: #6366f1;
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 8px;
}
.winner-emoji { font-size: 3.2rem; display: block; margin-bottom: 4px; }
.winner-genre { font-size: 2.6rem; font-weight: 800; color: #3730a3; }

/* ── Sidebar model badge ── */
.model-badge {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.85rem;
    color: #3730a3;
    font-weight: 600;
    margin-top: 6px;
    line-height: 1.6;
}
.badge-label {
    font-size: 0.70rem; text-transform: uppercase;
    letter-spacing: 0.07em; color: #6366f1; margin-bottom: 3px;
}

/* ── Section headers ── */
h3 { color: #1e1b4b !important; }
</style>
""", unsafe_allow_html=True)

# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and cache the SVM inference pipeline from disk."""
    missing = [p for p in [MODEL_PATH, SCALER_PATH, REDUCER_PATH] if not p.exists()]
    if missing:
        return None, None, None, None, None

    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    reducer = joblib.load(REDUCER_PATH)

    # Derive feature column order from the cache CSV header (authoritative).
    # Falls back to calling extract_features_from_y on a dummy signal.
    feat_cache = Path(FEAT_CACHE)
    if feat_cache.exists():
        header    = pd.read_csv(feat_cache, nrows=0)
        feat_cols = [c for c in header.columns if c not in ("label", "filename")]
    else:
        dummy = np.zeros(SR * 3, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat_dict = extract_features_from_y(dummy, sr=SR, n_mfcc=N_MFCC)
        feat_cols = list(feat_dict.keys())

    le = LabelEncoder()
    le.fit(GENRES)

    return model, scaler, reducer, feat_cols, le


@st.cache_data(show_spinner=False)
def load_audio_cached(file_bytes: bytes, suffix: str):
    """Decode uploaded audio bytes → numpy waveform (cached per file content)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=SR, mono=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return y, int(sr)


# ── Waveform plot ─────────────────────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int, n_segments: int = N_SEGMENTS):
    duration  = len(y) / sr
    seg_dur   = duration / n_segments

    # Downsample for fast rendering
    step   = max(1, len(y) // 12_000)
    t_plot = np.linspace(0, duration, len(y))[::step]
    y_plot = y[::step]

    amp_max = float(np.max(np.abs(y_plot))) or 1.0

    fig, ax = plt.subplots(figsize=(10, 2.5))

    # Alternating segment shading
    band_colors = ["#eef2ff", "#f8fafc"]
    for i in range(n_segments):
        ax.axvspan(
            i * seg_dur, (i + 1) * seg_dur,
            color=band_colors[i % 2], alpha=1.0, zorder=0,
        )

    # Segment boundary lines
    for i in range(1, n_segments):
        ax.axvline(i * seg_dur, color="#a5b4fc", linewidth=1.3,
                   linestyle="--", zorder=1)

    # Waveform line
    ax.plot(t_plot, y_plot, color="#4f46e5", linewidth=0.65, alpha=0.88, zorder=2)

    # Segment labels near the top
    for i in range(n_segments):
        mid = (i + 0.5) * seg_dur
        ax.text(
            mid, amp_max * 0.86, f"Seg {i + 1}",
            ha="center", va="top", fontsize=7.5, color="#6366f1",
            fontweight="bold", zorder=3,
        )

    ax.set_xlabel("Time (s)", fontsize=8.5, color="#6b7280")
    ax.set_ylabel("Amplitude", fontsize=8.5, color="#6b7280")
    ax.set_xlim(0, duration)
    ax.set_ylim(-amp_max * 1.05, amp_max * 1.05)
    ax.tick_params(labelsize=7.5, colors="#9ca3af")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#e5e7eb")
    ax.spines["bottom"].set_color("#e5e7eb")
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    plt.tight_layout(pad=0.4)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎵 Genre Classifier")
    st.markdown(
        "<div style='font-size:0.88rem; color:#6b7280; margin-top:-8px;'>"
        "Keerthan Shagrithaya</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Active Model**")
    st.markdown(
        """
        <div class="model-badge">
            <div class="badge-label">Algorithm · Features · Augmentation</div>
            SVM (RBF) &nbsp;·&nbsp; SelectKBest &nbsp;·&nbsp; no-aug
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        **Dataset:** GTZAN  
        **Genres:** 10 classes  
        **Features:** MFCC · Chroma · Spectral · Tonnetz  
        **Segments:** 5 equal slices per track  
        """
    )
    st.markdown("---")
    st.caption("CMPE 257 · Spring 2026 · SJSU")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 🎵 GTZAN Genre Classifier")
st.markdown(
    "Upload a music track — it will be split into **5 equal segments**, "
    "each classified independently, and the **majority-vote genre** is returned."
)

# ── How It Works ──────────────────────────────────────────────────────────────

with st.expander("✨ How it works", expanded=True):
    st.markdown("&nbsp;")
    steps = [
        ("📤", "Upload",   "Drop your audio track — wav, mp3, ogg, or flac"),
        ("✂️",  "Slice",    "Track is divided into 5 equal time segments"),
        ("🔬", "Extract",  "166 hand-crafted audio features pulled per segment"),
        ("🤖", "Predict",  "SVM (RBF) model classifies each segment's genre"),
        ("🗳️", "Vote",     "The genre predicted most often wins"),
    ]
    cols = st.columns(5, gap="small")
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-icon">{icon}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("&nbsp;")

st.markdown("---")

# ── File upload ───────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a music track",
    type=["wav", "mp3", "ogg", "flac"],
    help="Supports wav, mp3, ogg, flac. Longer tracks (≥ 30 s) give the best results.",
)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    suffix     = Path(uploaded.name).suffix or ".wav"

    # Audio playback
    st.audio(uploaded)

    # Load audio (cached by file content — no re-load on re-run)
    with st.spinner("Decoding audio..."):
        y, sr_loaded = load_audio_cached(file_bytes, suffix)

    duration_s   = len(y) / sr_loaded
    total_needed = N_SEGMENTS * SEG_DURATION   # 150 s

    if duration_s < SEG_DURATION:
        st.warning(
            f"⚠️ Track is only {duration_s:.1f}s — shorter than one 30s segment. "
            "Accuracy may be reduced."
        )
    elif duration_s < total_needed:
        st.warning(
            f"⚠️ Track is {duration_s:.1f}s (less than {total_needed}s needed for "
            f"5 × 30s segments). Later segments will be zero-padded."
        )

    # Waveform: show only the first 150 s (the region that will be analysed)
    y_display  = y[: int(total_needed * sr_loaded)]
    display_s  = len(y_display) / sr_loaded

    st.markdown("#### 📊 Waveform — 5 × 30s Segments")
    st.caption(
        f"Showing first {display_s:.1f}s of {duration_s:.1f}s &nbsp;·&nbsp; "
        f"Each segment: 30s &nbsp;·&nbsp; "
        f"Sample rate: {sr_loaded:,} Hz"
    )
    waveform_fig = plot_waveform(y_display, sr_loaded)
    st.pyplot(waveform_fig, use_container_width=True)
    plt.close(waveform_fig)

    st.markdown("---")
    analyze = st.button(
        "🔍 Analyze Genre",
        type="primary",
        use_container_width=True,
    )

    # ── Analysis ──────────────────────────────────────────────────────────────

    if analyze:
        # Pre-flight model check
        missing_files = [
            p.name for p in [MODEL_PATH, SCALER_PATH, REDUCER_PATH]
            if not p.exists()
        ]
        if missing_files:
            st.error(
                f"Missing model file(s): {', '.join(missing_files)}. "
                "Run `main.py` to train and save models first."
            )
            st.stop()

        predictions = []
        seg_info    = []   # list of (start_s, end_s, genre)

        with st.status("🔍 Analyzing your track...", expanded=True) as status:

            # Step 1 — Load pipeline
            st.write("⏳ Loading model pipeline...")
            model, scaler, reducer, feat_cols, le = load_pipeline()
            st.write("✅ Model loaded — SVM (RBF) · SelectKBest (k=80) · no-aug")

            # Step 2 — Confirm audio
            st.write("⏳ Preparing audio...")
            use_s = min(duration_s, N_SEGMENTS * SEG_DURATION)
            st.write(
                f"✅ Audio ready — using first {use_s:.1f}s of {duration_s:.1f}s "
                f"at {sr_loaded:,} Hz"
            )

            # Step 3 — Slice into 5 × 30s windows (zero-pad last if needed)
            st.write(f"⏳ Slicing into {N_SEGMENTS} × {SEG_DURATION}s segments...")
            segments = []
            for i in range(N_SEGMENTS):
                start = i * SEG_SAMPLES
                chunk = y[start : start + SEG_SAMPLES]
                if len(chunk) < SEG_SAMPLES:
                    chunk = np.pad(chunk, (0, SEG_SAMPLES - len(chunk)))
                segments.append(chunk)
            st.write(
                f"✅ Track sliced — {N_SEGMENTS} segments of {SEG_DURATION}s each"
            )

            # Step 4 — Per-segment inference
            progress_bar = st.progress(0, text="Starting segment analysis...")

            for i, seg in enumerate(segments):
                seg_start = i * SEG_DURATION
                seg_end   = (i + 1) * SEG_DURATION

                st.write(f"⏳ Segment {i + 1}/{N_SEGMENTS} — extracting audio features...")
                feat_dict = extract_features_from_y(seg, sr=SR, n_mfcc=N_MFCC)
                feat_vec  = np.array([[feat_dict[c] for c in feat_cols]])

                st.write(f"⏳ Segment {i + 1}/{N_SEGMENTS} — running inference...")
                X_s      = scaler.transform(feat_vec)
                X_r      = reducer.transform(X_s) if reducer is not None else X_s
                pred_int = model.predict(X_r)[0]
                genre    = le.inverse_transform([pred_int])[0]

                predictions.append(genre)
                seg_info.append((seg_start, seg_end, genre))

                emoji = GENRE_EMOJI.get(genre, "🎵")
                st.write(
                    f"✅ Segment {i + 1}/{N_SEGMENTS} "
                    f"({seg_start:.1f}s – {seg_end:.1f}s) "
                    f"→ **{genre.capitalize()}** {emoji}"
                )
                progress_bar.progress(
                    (i + 1) / N_SEGMENTS,
                    text=f"Segment {i + 1}/{N_SEGMENTS} complete",
                )

            status.update(
                label="✅ Analysis complete!",
                state="complete",
                expanded=False,
            )

        # ── Segment result cards ───────────────────────────────────────────────

        st.markdown("### 🎯 Segment Predictions")
        seg_cols = st.columns(N_SEGMENTS, gap="small")
        for idx, (col, (start_s, end_s, genre)) in enumerate(
            zip(seg_cols, seg_info), start=1
        ):
            emoji         = GENRE_EMOJI.get(genre, "🎵")
            bg, border    = GENRE_COLORS.get(genre, ("#f9fafb", "#d1d5db"))
            with col:
                st.markdown(
                    f"""
                    <div class="seg-card"
                         style="background:{bg}; border-color:{border};">
                        <div class="seg-num">Segment {idx}</div>
                        <div class="seg-time">{start_s:.1f}s – {end_s:.1f}s</div>
                        <span class="seg-emoji">{emoji}</span>
                        <div class="seg-genre">{genre.capitalize()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Vote distribution chart ────────────────────────────────────────────

        st.markdown("### 🗳️ Vote Distribution")
        vote_counts = Counter(predictions)
        vote_df = (
            pd.DataFrame(
                {
                    "Genre": [g.capitalize() for g in GENRES],
                    "Votes": [vote_counts.get(g, 0) for g in GENRES],
                }
            )
            .set_index("Genre")
            .query("Votes > 0")
        )
        st.bar_chart(vote_df, use_container_width=True, color="#4f46e5")

        # ── Winner banner ──────────────────────────────────────────────────────

        winner       = Counter(predictions).most_common(1)[0][0]
        winner_emoji = GENRE_EMOJI.get(winner, "🎵")
        winner_votes = vote_counts[winner]

        st.markdown(
            f"""
            <div class="winner-banner">
                <div class="winner-label">🏆 Majority Vote — Predicted Genre</div>
                <span class="winner-emoji">{winner_emoji}</span>
                <div class="winner-genre">{winner.capitalize()}</div>
                <div style="margin-top:10px; font-size:0.88rem; color:#6366f1;">
                    {winner_votes} out of {N_SEGMENTS} segments agreed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
